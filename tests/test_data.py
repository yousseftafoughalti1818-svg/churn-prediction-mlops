from churn.data import clean, split_xy


def test_total_charges_blank_becomes_zero(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()


def test_target_is_binary(raw_df):
    df = clean(raw_df)
    assert set(df["Churn"].unique()) <= {0, 1}


def test_split_drops_id_and_target(raw_df):
    X, y = split_xy(clean(raw_df))
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert len(X) == len(y)
