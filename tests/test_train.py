import json

from churn import config
from churn.data import clean, split_xy
from churn.predict import load_model, predict
from churn.train import train


def test_training_end_to_end(tmp_project, raw_df):
    load_model.cache_clear()
    metrics = train(config.RAW_DATA_PATH)

    assert config.MODEL_PATH.exists()
    assert json.loads(config.METRICS_PATH.read_text())["best_model"] == metrics["best_model"]
    assert 0.5 < metrics["models"][metrics["best_model"]]["test_roc_auc"] <= 1.0

    artifact = load_model(config.MODEL_PATH)
    X, _ = split_xy(clean(raw_df))
    preds = predict(X.head(5), artifact)
    assert preds["churn_probability"].between(0, 1).all()
    assert set(preds["risk_level"]) <= {"low", "medium", "high"}
    load_model.cache_clear()
