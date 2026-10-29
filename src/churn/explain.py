"""Model explainability: global feature importance and per-customer explanations.

SHAP is used when installed (TreeExplainer for tree models, LinearExplainer
for logistic regression). Otherwise the module falls back to sklearn's
permutation importance so the project still runs.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from churn import config

log = logging.getLogger("churn.explain")


def _transformed(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocess"]
    Xt = pre.transform(X)
    names = pre.named_steps["columns"].get_feature_names_out()
    return pd.DataFrame(Xt, columns=names, index=X.index)


def _shap_values(pipeline, Xt: pd.DataFrame) -> np.ndarray:
    import shap

    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[..., 1]
    return values


def explain_model(pipeline, X: pd.DataFrame, y: pd.Series | None = None, top_n: int = 15) -> dict:
    """Compute global importance, save a bar chart and a JSON summary."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Xt = _transformed(pipeline, X)

    try:
        values = _shap_values(pipeline, Xt)
        importance = pd.Series(np.abs(values).mean(axis=0), index=Xt.columns)
        method = "mean |SHAP value|"
        try:
            import shap

            shap.summary_plot(values, Xt, show=False, max_display=top_n)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=120)
            plt.close()
        except Exception as exc:
            log.warning("SHAP summary plot skipped: %s", exc)
    except ImportError:
        log.warning("shap not installed: using permutation importance.")
        if y is None:
            raise ValueError("permutation importance needs the true labels y")
        result = permutation_importance(
            pipeline, X, y, scoring="roc_auc", n_repeats=5,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        importance = pd.Series(result.importances_mean, index=X.columns)
        method = "permutation importance"

    importance = importance.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    importance[::-1].plot.barh(ax=ax, color="#1f77b4")
    ax.set(title=f"Top {top_n} churn drivers ({method})", xlabel=method)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    summary = {"method": method, "top_features": importance.round(4).to_dict()}
    (config.REPORTS_DIR / "feature_importance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def explain_customer(pipeline, customer: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Return the features that push one customer towards (or away from) churn."""
    Xt = _transformed(pipeline, customer)
    try:
        values = _shap_values(pipeline, Xt)[0]
    except ImportError:
        return []
    order = np.argsort(-np.abs(values))[:top_n]
    return [
        {"feature": Xt.columns[i], "impact": round(float(values[i]), 4)} for i in order
    ]
