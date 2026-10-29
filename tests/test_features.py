import numpy as np

from churn import config
from churn.data import clean, split_xy
from churn.features import add_features, build_preprocessor


def test_engineered_columns_exist(raw_df):
    out = add_features(clean(raw_df))
    for col in config.ENGINEERED_NUMERIC + config.ENGINEERED_CATEGORICAL:
        assert col in out.columns


def test_num_services_range(raw_df):
    out = add_features(clean(raw_df))
    assert out["num_services"].between(0, 9).all()


def test_month_to_month_flag(raw_df):
    out = add_features(clean(raw_df))
    expected = (out["Contract"] == "Month-to-month").astype(int)
    assert (out["is_month_to_month"] == expected).all()


def test_preprocessor_output_is_finite(raw_df):
    X, _ = split_xy(clean(raw_df))
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert np.isfinite(Xt).all()
