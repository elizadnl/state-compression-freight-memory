#!/usr/bin/env python3
"""Fetch and verify the public source file, then prepare the 808-row BDI analysis CSV.

This repository deliberately does not redistribute the raw BDI index levels.
The source file is publicly accessible from the Zenodo record cited in the paper.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import io
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public_bdi_2009_2025.csv"
URL = "https://zenodo.org/records/22035531/files/predictors_FINAL.csv?download=1"
SOURCE_MD5 = "f7f7c992711c9ee013c2b0799aa9a6d4"  # published on Zenodo record
OUTPUT_SHA256 = "10210ad581160b344dbde9ce5ef94424815fc8f3910db3d57204ff3a5fc894e0"


def digest(data: bytes, algo: str) -> str:
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def main():
    manual = Path(__file__).resolve().parent / "predictors_FINAL.csv"
    if manual.exists():
        print(f"Using manually downloaded source: {manual}")
        raw = manual.read_bytes()
    else:
        print(f"Fetching public source: {URL}")
        req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=60) as r:
                raw = r.read()
        except Exception as exc:
            raise SystemExit(
                "Automatic download failed. Download predictors_FINAL.csv manually from "
                "https://zenodo.org/records/22035531 and place it at "
                "src/predictors_FINAL.csv, then rerun. Original error: " + str(exc)
            )

    md5 = digest(raw, "md5")
    if md5 != SOURCE_MD5:
        raise SystemExit(f"Source MD5 mismatch: expected {SOURCE_MD5}, got {md5}")

    src = pd.read_csv(io.BytesIO(raw))
    required = {"date", "bdry"}
    if not required.issubset(src.columns):
        raise SystemExit(f"Expected columns {sorted(required)}, got {list(src.columns)}")

    d = src[["date", "bdry"]].rename(columns={"bdry": "bdi"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) != 808:
        raise SystemExit(f"Expected 808 rows, got {len(d)}")
    if d.date.iloc[0] != pd.Timestamp("2009-11-06") or d.date.iloc[-1] != pd.Timestamp("2025-04-25"):
        raise SystemExit("Unexpected sample endpoints")
    if d.bdi.isna().any() or d.date.duplicated().any():
        raise SystemExit("Missing BDI values or duplicate dates found")
    if set(d.date.diff().dropna().dt.days.unique()) != {7}:
        raise SystemExit("Weekly spacing check failed")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    got = digest(OUT.read_bytes(), "sha256")
    if got != OUTPUT_SHA256:
        raise SystemExit(f"Prepared-file SHA256 mismatch: expected {OUTPUT_SHA256}, got {got}")
    print(f"Prepared {OUT} ({len(d)} rows); SHA256 verified.")


if __name__ == "__main__":
    main()
