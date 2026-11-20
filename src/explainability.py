import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.train_and_evaluate import (
    FIG_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    SEED,
    engineer_features,
    load_data,
    positive_class_shap,
    stratified_split,
)


def get_base_value(explainer: shap.TreeExplainer) -> float:
    return float(np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))[-1])


class RiskExplainer:
    def __init__(self, model=None):
        self.model = model if model is not None else joblib.load(MODELS_DIR / "best_model.joblib")
        with open(MODELS_DIR / "feature_list.json") as f:
            self.features = json.load(f)
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = get_base_value(self.explainer)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return positive_class_shap(self.explainer, X[self.features])

    def top_factors(self, X_row: pd.DataFrame, k: int = 5) -> list:
        values = self.shap_values(X_row)[0]
        order = np.argsort(-np.abs(values))[:k]
        return [
            {
                "feature": self.features[i],
                "value": float(X_row.iloc[0][self.features[i]]),
                "shap_value": float(values[i]),
                "direction": "increases risk" if values[i] > 0 else "decreases risk",
            }
            for i in order
        ]

    def explanation(self, X_row: pd.DataFrame) -> shap.Explanation:
        values = self.shap_values(X_row)[0]
        return shap.Explanation(
            values=values,
            base_values=self.base_value,
            data=X_row[self.features].iloc[0].to_numpy(dtype=float),
            feature_names=self.features,
        )


def save_global_plots(svc: RiskExplainer, X_test: pd.DataFrame, n_samples: int = 3000) -> None:
    sample = X_test[svc.features].sample(n=min(n_samples, len(X_test)), random_state=SEED)
    values = svc.shap_values(sample)

    plt.figure(figsize=(9, 8))
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot (Positive Class: Diabetes Risk)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, sample, plot_type="bar", max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_bar_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame({"Feature": svc.features, "Mean_Abs_SHAP": np.abs(values).mean(axis=0)})
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .round(5)
    )
    importance.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
    print("Top global drivers:")
    print(importance.head(10).to_string(index=False))


def save_waterfall(svc: RiskExplainer, X_row: pd.DataFrame, probability: float, title: str, path) -> None:
    shap.plots.waterfall(svc.explanation(X_row), max_display=12, show=False)
    plt.title(f"{title} (predicted probability {probability:.2f})", fontsize=11)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")


def save_local_plots(svc: RiskExplainer, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    Xte = X_test[svc.features].reset_index(drop=True)
    labels = y_test.reset_index(drop=True).to_numpy()
    proba = svc.model.predict_proba(Xte)[:, 1]

    positives = np.where(labels == 1)[0]
    cases = {
        "waterfall_high_risk.png": ("Highest-risk test patient", int(np.argmax(proba))),
        "waterfall_low_risk.png": ("Lowest-risk test patient", int(np.argmin(proba))),
        "waterfall_median_positive.png": (
            "Median-risk true positive",
            int(positives[np.argsort(proba[positives])[len(positives) // 2]]),
        ),
    }
    for filename, (title, idx) in cases.items():
        save_waterfall(svc, Xte.iloc[[idx]], float(proba[idx]), title, FIG_DIR / filename)
        print(f"Saved {filename}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    _, _, X_test, _, _, y_test = stratified_split(df)
    Xte = engineer_features(X_test)
    svc = RiskExplainer()
    save_global_plots(svc, Xte)
    save_local_plots(svc, Xte, y_test)
    print(f"\nExplainability figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()