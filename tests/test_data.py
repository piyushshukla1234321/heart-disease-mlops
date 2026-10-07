import numpy as np
import pandas as pd
import pytest

from cardiorisk import config as C
from cardiorisk.data import _validate_raw_bytes, clean_data, load_raw, summarize


def test_load_raw_parses_question_marks_as_missing(raw_file):
    df = load_raw(raw_file)
    assert list(df.columns) == C.RAW_COLUMNS
    assert np.isnan(df.loc[3, "ca"]) and np.isnan(df.loc[7, "thal"])


def test_clean_output_schema(clean_df):
    assert list(clean_df.columns) == C.FEATURES + [C.TARGET]
    assert set(clean_df[C.TARGET].unique()) <= {0, 1}


def test_target_is_binarised_from_num(raw_df):
    clean = clean_data(raw_df)
    assert clean[C.TARGET].sum() == (raw_df["num"] > 0).sum()


def test_out_of_range_and_unknown_codes_become_missing(raw_df):
    raw = raw_df.copy()
    raw.loc[0, "chol"] = 0        # physiologically impossible
    raw.loc[1, "age"] = 250
    raw.loc[2, "cp"] = 9          # not a valid chest-pain code
    clean = clean_data(raw)
    assert pd.isna(clean.loc[0, "chol"])
    assert pd.isna(clean.loc[1, "age"])
    assert pd.isna(clean.loc[2, "cp"])


def test_rows_without_target_dropped_and_duplicates_removed(raw_df):
    raw = pd.concat([raw_df, raw_df.iloc[[0, 0]]], ignore_index=True)
    raw.loc[5, "num"] = np.nan
    clean = clean_data(raw)
    assert len(clean) == len(raw_df) - 1
    assert not clean.duplicated().any()


def test_clean_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        clean_data(raw_df.drop(columns=["thal"]))


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_summary_counts_are_consistent(raw_df):
    clean = clean_data(raw_df)
    s = summarize(raw_df, clean)
    assert sum(s["class_counts"].values()) == s["clean_rows"] == len(clean)
    assert 0 < s["positive_rate"] < 1


def test_validate_raw_bytes_rejects_bad_payload():
    with pytest.raises(ValueError):
        _validate_raw_bytes(b"<html>not the dataset</html>")
