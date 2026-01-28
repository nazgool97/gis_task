import argparse
import os
from datetime import timezone
from io import BytesIO, StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv

from arcgis.gis import GIS

VALUE_COLS = [f"Значення {i}" for i in range(1, 11)]
BASE_COLS = ["Дата", "Область", "Місто", "long", "lat"]


def load_sheet_csv(sheet_id: str, gid: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    head = response.text[:500].lower()
    if "<html" in head or "accounts.google.com" in head:
        raise ValueError(
            "Google Sheets returned HTML instead of CSV. "
            "Check that the sheet is shared for viewing and that SHEET_ID/SHEET_GID are correct."
        )
    try:
        return pd.read_csv(BytesIO(response.content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(response.content), encoding="cp1251")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in BASE_COLS + VALUE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns: {missing}. Found columns: {found}".format(
                missing=missing, found=df.columns.tolist()
            )
        )

    for col in VALUE_COLS:
        series = pd.to_numeric(df[col], errors="coerce")
        df[col] = series.fillna(0).clip(lower=0).astype(int)

    df["long"] = df["long"].apply(parse_coord)
    df["lat"] = df["lat"].apply(parse_coord)
    return df


def parse_coord(value) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", ".")
    return float(text)


def expand_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        counts = [int(row[c]) for c in VALUE_COLS]
        max_count = max(counts) if counts else 0
        if max_count <= 0:
            continue

        base = {col: row[col] for col in BASE_COLS}
        for i in range(max_count):
            new_row = dict(base)
            for col, count in zip(VALUE_COLS, counts):
                new_row[col] = 1 if i < count else 0
            rows.append(new_row)
    return pd.DataFrame(rows, columns=BASE_COLS + VALUE_COLS)


def parse_date_to_epoch_ms(value) -> Optional[int]:
    if pd.isna(value):
        return None
    dt = date_parser.parse(str(value), dayfirst=True)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def build_features(df: pd.DataFrame) -> List[Dict]:
    features = []
    for _, row in df.iterrows():
        attrs = {
            "d_date": parse_date_to_epoch_ms(row["Дата"]),
            "t_region": row["Область"],
            "t_city": row["Місто"],
        }
        for i in range(1, 11):
            attrs[f"i_value_{i}"] = int(row[f"Значення {i}"])

        attrs["long"] = float(row["long"])
        attrs["lat"] = float(row["lat"])

        features.append(
            {
                "attributes": attrs,
                "geometry": {
                    "x": float(row["long"]),
                    "y": float(row["lat"]),
                    "spatialReference": {"wkid": 4326},
                },
            }
        )
    return features


def chunk_list(items: List[Dict], size: int) -> List[List[Dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Update ArcGIS Online layer from Google Sheets")
    parser.add_argument("--sheet-id", default=os.getenv("SHEET_ID"))
    parser.add_argument("--sheet-gid", default=os.getenv("SHEET_GID"))
    parser.add_argument("--arcgis-url", default=os.getenv("ARCGIS_URL", "https://www.arcgis.com"))
    parser.add_argument("--arcgis-username", default=os.getenv("ARCGIS_USERNAME"))
    parser.add_argument("--arcgis-password", default=os.getenv("ARCGIS_PASSWORD"))
    parser.add_argument("--item-id", default=os.getenv("ARCGIS_ITEM_ID"))
    parser.add_argument("--truncate", action="store_true", default=os.getenv("TRUNCATE_BEFORE_LOAD", "").lower() == "true")
    args = parser.parse_args()

    required = [
        ("sheet-id", args.sheet_id),
        ("sheet-gid", args.sheet_gid),
        ("arcgis-username", args.arcgis_username),
        ("arcgis-password", args.arcgis_password),
        ("item-id", args.item_id),
    ]
    missing = [name for name, value in required if not value]
    if missing:
        raise SystemExit(f"Missing required arguments: {missing}")

    raw_df = load_sheet_csv(args.sheet_id, args.sheet_gid)
    df = normalize_dataframe(raw_df)
    expanded = expand_rows(df)
    features = build_features(expanded)

    gis = GIS(args.arcgis_url, args.arcgis_username, args.arcgis_password)
    item = gis.content.get(args.item_id)
    if not item:
        raise SystemExit("ArcGIS item not found.")

    _ = item.resources.list()

    layer = item.layers[0]
    if args.truncate:
        layer.delete_features(where="1=1")

    for chunk in chunk_list(features, 500):
        result = layer.edit_features(adds=chunk)
        if not result or "addResults" not in result:
            raise SystemExit(f"Failed to append features: {result}")

    print(f"Added {len(features)} features.")


if __name__ == "__main__":
    main()
