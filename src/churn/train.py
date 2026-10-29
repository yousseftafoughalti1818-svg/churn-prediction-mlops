"""Train, compare and save churn models.

Usage:
    python -m churn.train
"""
from __future__ import annotations

import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from churn import config
from churn.data import clean, load_raw, split_xy
from churn.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("churn.train")


def get_models(y_train: pd.Series) -> dict:
    """Candidate models. Class imbalance is handled with class weights."""
    neg, pos = np.bincount(y_train)
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:  # keeps the pipeline usable without xgboost installed
        from sklearn.ensemble import HistGradientBoostingClassifier

        log.warning("xgboost not installed: using HistGradientBoosting instead.")
        models["gradient_boosting"] = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300,
            class_weight="balanced", random_state=config.RANDOM_STATE,
        )
    return models


def make_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Decision threshold that maximises F1 (computed on out-of-fold predictions)."""
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def lift_at_top(y_true: np.ndarray, proba: np.ndarray, top: float = 0.10) -> float:
    """Churn rate among the top x% riskiest customers divided by the global churn rate."""
    n = max(1, int(len(proba) * top))
    idx = np.argsort(-proba)[:n]
    return float(y_true[idx].mean() / y_true.mean())


def business_rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """'Existing approach': flag every month-to-month customer as at risk."""
    return X["Contract"].eq("Month-to-month").astype(int).to_numpy()


def save_figures(y_test, probas: dict, best_name: str, best_pred) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def write_markdown_report(metrics: dict) -> None:
    lines = [
        "# Model results (test set)",
        "",
        f"Best model: **{metrics['best_model']}** "
        f"(decision threshold = {metrics['threshold']:.2f})",
        "",
        "| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |",
        "|---|---|---|---|",
    ]
    for name, m in metrics["models"].items():
        lines.append(
            f"| {name} | {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f} "
            f"| {m['test_roc_auc']:.3f} | {m['test_pr_auc']:.3f} |"
        )
    b, r = metrics["best_model_test"], metrics["baseline_rule"]
    lines += [
        "",
        "| | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| Best model | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |",
        f"| Business rule (month-to-month) | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |",
        "",
        f"- F1 improvement vs business rule: **{metrics['f1_improvement_vs_rule_pct']:+.1f}%**",
        f"- Lift in the top 10% riskiest customers: **x{b['lift_top10']:.2f}**",
    ]
    (config.REPORTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(data_path=None) -> dict:
    df = clean(load_raw(data_path))
    X, y = split_xy(df)
    log.info("Dataset: %d customers, churn rate = %.1f%%", len(df), 100 * y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted, test_probas, oof_probas = {}, {}, {}, {}
    for name, model in get_models(y_train).items():
        pipe = make_pipeline(model)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        fold_aucs = [
            roc_auc_score(y_train.iloc[va], oof[va]) for _, va in cv.split(X_train, y_train)
        ]
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_roc_auc_mean": float(np.mean(fold_aucs)),
            "cv_roc_auc_std": float(np.std(fold_aucs)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
            "test_pr_auc": float(average_precision_score(y_test, proba)),
        }
        fitted[name], test_probas[name], oof_probas[name] = pipe, proba, oof
        log.info("%-20s CV AUC=%.3f  test AUC=%.3f", name,
                 results[name]["cv_roc_auc_mean"], results[name]["test_roc_auc"])

    # Model selection is done on cross-validation only (no test-set leakage)
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    threshold = best_f1_threshold(y_train.to_numpy(), oof_probas[best_name])
    best_proba = test_probas[best_name]
    best_pred = (best_proba >= threshold).astype(int)

    rule_pred = business_rule_baseline(X_test)
    best_metrics = {
        "precision": float(precision_score(y_test, best_pred)),
        "recall": float(recall_score(y_test, best_pred)),
        "f1": float(f1_score(y_test, best_pred)),
        "lift_top10": lift_at_top(y_test.to_numpy(), best_proba),
    }
    rule_metrics = {
        "precision": float(precision_score(y_test, rule_pred)),
        "recall": float(recall_score(y_test, rule_pred)),
        "f1": float(f1_score(y_test, rule_pred)),
    }
    metrics = {
        "n_customers": int(len(df)),
        "churn_rate": float(y.mean()),
        "models": results,
        "best_model": best_name,
        "threshold": threshold,
        "best_model_test": best_metrics,
        "baseline_rule": rule_metrics,
        "f1_improvement_vs_rule_pct": 100 * (best_metrics["f1"] / rule_metrics["f1"] - 1),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    final_model = fitted[best_name]
    joblib.dump(
        {"pipeline": final_model, "threshold": threshold, "model_name": best_name},
        config.MODEL_PATH,
    )
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_figures(y_test, test_probas, best_name, best_pred)
    write_markdown_report(metrics)

    # Feature importance / explainability (SHAP when available)
    try:
        from churn.explain import explain_model

        explain_model(final_model, X_test, y_test)
    except Exception as exc:  # explainability must never break training
        log.warning("Explainability step skipped: %s", exc)

    log.info("Best model: %s | threshold=%.2f | F1=%.3f | recall=%.3f",
             best_name, threshold, best_metrics["f1"], best_metrics["recall"])
    log.info("Model saved to %s", config.MODEL_PATH)
    return metrics


if __name__ == "__main__":
    train()
