"""Download and clean the UCI Heart Disease (Cleveland) dataset.

Run from the repository root:
    python -m cardiorisk.data            # download if missing, then clean
    python -m cardiorisk.data --force    # re-download the raw file first
"""
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cardiorisk import config as C

log = logging.getLogger(__name__)


# ------------------------------------------------------------ acquisition
def download_raw(dest: Path = C.RAW_FILE, force: bool = False) -> Path:
    """Fetch processed.cleveland.data from UCI. Skips the download if the file exists."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Raw file already present, skipping download: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in C.UCI_URLS:
        try:
            log.info("Downloading %s", url)
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = resp.read()
            if url.endswith(".zip"):
                payload = _extract_from_zip(payload, dest.name)
            _validate_raw_bytes(payload)
            dest.write_bytes(payload)
            log.info("Saved raw data to %s (%d bytes)", dest, len(payload))
            return dest
        except Exception as exc:  # try the next mirror
            errors.append(f"  {url} -> {exc}")

    raise RuntimeError(
        "Could not download the dataset from any source:\n" + "\n".join(errors)
        + f"\nManual fix: download 'processed.cleveland.data' from the UCI Heart Disease "
        f"page and place it at {dest}"
    )


def _extract_from_zip(payload: bytes, member_name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        matches = [n for n in zf.namelist() if n.endswith(member_name)]
        if not matches:
            raise ValueError(f"{member_name} not found inside the zip archive")
        return zf.read(matches[0])


def _validate_raw_bytes(payload: bytes) -> None:
    lines = [ln for ln in payload.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    if len(lines) < 250:
        raise ValueError(f"expected ~303 rows, got {len(lines)}")
    n_fields = len(lines[0].split(","))
    if n_fields != len(C.RAW_COLUMNS):
        raise ValueError(f"expected {len(C.RAW_COLUMNS)} comma-separated fields, got {n_fields}")


# --------------------------------------------------------------- loading
def load_raw(path: Path = C.RAW_FILE) -> pd.DataFrame:
    """Read the header-less raw file. UCI marks missing values with '?'."""
    return pd.read_csv(path, header=None, names=C.RAW_COLUMNS, na_values=["?"],
                       skipinitialspace=True)


# -------------------------------------------------------------- cleaning
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with FEATURES + binary TARGET.

    What cleaning does (and deliberately does not do):
      * coerces every column to numeric; anything unparseable becomes NaN
      * drops rows with no target, and binarises num (0 -> 0, 1..4 -> 1)
      * turns physiologically impossible values and unknown category codes into NaN
      * removes exact duplicate rows
      * does NOT impute. Imputation lives inside the model pipeline so it is
        learned from training rows only (no test-set leakage) and is applied
        identically at serving time.
    """
    missing = set(C.RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")

    df = raw[C.RAW_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.dropna(subset=[C.RAW_TARGET]).copy()
    df[C.TARGET] = (df[C.RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=[C.RAW_TARGET])

    for col, (lo, hi) in C.VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan
    for col, codes in C.ALLOWED_CODES.items():
        bad = df[col].notna() & ~df[col].isin(codes)
        df.loc[bad, col] = np.nan

    df = df.drop_duplicates().reset_index(drop=True)
    return df[C.FEATURES + [C.TARGET]]


def summarize(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Numbers worth printing / putting in the report."""
    counts = clean[C.TARGET].value_counts().sort_index()
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "rows_removed": int(len(raw) - len(clean)),
        "missing_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "missing_clean": {k: int(v) for k, v in clean.isna().sum().items() if v},
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "positive_rate": round(float(clean[C.TARGET].mean()), 4),
    }


def prepare_dataset(force_download: bool = False) -> pd.DataFrame:
    """Full acquisition step: download (if needed) -> clean -> save CSV."""
    raw = load_raw(download_raw(force=force_download))
    clean = clean_data(raw)
    C.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(C.CLEAN_FILE, index=False)
    log.info("Saved cleaned data to %s", C.CLEAN_FILE)
    for key, value in summarize(raw, clean).items():
        log.info("  %-14s %s", key, value)
    return clean


def load_clean(path: Path = C.CLEAN_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m cardiorisk.data` first")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download the raw file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_dataset(force_download=args.force)


if __name__ == "__main__":
    main()
