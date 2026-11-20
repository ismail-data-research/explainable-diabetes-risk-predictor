import json
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

TARGET = "Diabetes_binary"
RAW_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
ORDINAL_FEATURES = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
BINARY_FEATURES = [f for f in RAW_FEATURES if f not in ORDINAL_FEATURES]
ENGINEERED_FEATURES = ["Health_Risk_Score", "BMI_HighBP_Interaction", "Lifestyle_Risk_Score"]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Health_Risk_Score"] = (
        out["HighBP"] + out["HighChol"] + out["Smoker"] + out["HeartDiseaseorAttack"]
    )
    out["BMI_HighBP_Interaction"] = out["BMI"] * out["HighBP"]
    out["Lifestyle_Risk_Score"] = (
        (1 - out["PhysActivity"]) + (1 - out["Fruits"]) + (1 - out["Veggies"]) + out["HvyAlcoholConsump"]
    )
    return out[FEATURES].astype(float)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download "
            "'diabetes_binary_health_indicators_BRFSS2015.csv' from Kaggle into the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    before = len(df)
    df = df[RAW_FEATURES + [TARGET]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Rows loaded: {before:,} | after dropping duplicates/NaN: {len(df):,}")
    print(f"Positive class rate: {df[TARGET].mean():.4f}")
    return df


def stratified_split(df: pd.DataFrame):
    X = df[RAW_FEATURES].astype(float)
    y = df[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    cat_idx = [RAW_FEATURES.index(c) for c in BINARY_FEATURES]
    sampler = SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=RAW_FEATURES).astype(float)
    for col in ORDINAL_FEATURES:
        X_res[col] = X_res[col].round()
    y_res = pd.Series(y_res, name=TARGET).astype(int)
    print(f"After SMOTENC (train only): {len(X_res):,} rows | positive rate {y_res.mean():.4f}")
    return X_res, y_res


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=5,
            n_jobs=-1, random_state=SEED,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="auc", early_stopping_rounds=50,
            n_jobs=-1, random_state=SEED,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=-1, random_state=SEED, verbose=-1,
        ),
    }


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
    }


def plot_roc(y_test, test_probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y_test, test_probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in test_probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP = {average_precision_score(y_test, proba):.3f})")
    ax.axhline(float(np.mean(y_test)), color="k", ls="--", lw=1, label="Prevalence baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves on Held-Out Test Set")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_test, proba, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= 0.5).astype(int), ax=ax, cmap="Blues", colorbar=False,
        display_labels=["No Diabetes", "Prediabetes/Diabetes"],
    )
    ax.set_title(f"Confusion Matrix: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def positive_class_shap(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    elif getattr(values, "ndim", 2) == 3:
        values = values[:, :, 1]
    return np.asarray(values)


def plot_shap_summary(model, X_test: pd.DataFrame, path: Path, n_samples: int = 2000) -> None:
    sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    values = positive_class_shap(explainer, sample)
    plt.figure(figsize=(9, 8))
    shap.summary_plot(values, sample, max_display=15, show=False)
    plt.title("SHAP Summary Plot (Positive Class: Diabetes Risk)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    Xtr_res = engineer_features(X_train_res)
    Xtr_orig = engineer_features(X_train)
    Xva = engineer_features(X_val)
    Xte = engineer_features(X_test)

    scaler = StandardScaler().fit(Xtr_orig)
    Xtr_res_s = scaler.transform(Xtr_res)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    models = build_models()
    rows, test_probas, val_scores, fitted = [], {}, {}, {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        if name == "Logistic Regression":
            model.fit(Xtr_res_s, y_train_res)
            val_proba = model.predict_proba(Xva_s)[:, 1]
            test_proba = model.predict_proba(Xte_s)[:, 1]
        else:
            if name == "XGBoost":
                model.fit(Xtr_res, y_train_res, eval_set=[(Xva, y_val)], verbose=False)
            elif name == "LightGBM":
                model.fit(
                    Xtr_res, y_train_res, eval_set=[(Xva, y_val)], eval_metric="auc",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                model.fit(Xtr_res, y_train_res)
            val_proba = model.predict_proba(Xva)[:, 1]
            test_proba = model.predict_proba(Xte)[:, 1]

        val_scores[name] = average_precision_score(y_val, val_proba)
        test_probas[name] = test_proba
        fitted[name] = model
        metrics = compute_metrics(y_test, test_proba)
        metrics["Val_PR_AUC"] = val_scores[name]
        metrics["Train_Time_s"] = time.time() - start
        rows.append({"Model": name, **metrics})
        print(f"{name} done in {metrics['Train_Time_s']:.1f}s | Test ROC-AUC {metrics['ROC_AUC']:.4f}")

    results = pd.DataFrame(rows).round(4)
    results.to_csv(OUTPUT_DIR / "model_benchmark_results.csv", index=False)
    print("\n=== Benchmark (held-out test set) ===")
    print(results.to_string(index=False))

    best_name = max(TREE_MODELS, key=lambda n: val_scores[n])
    best_model = fitted[best_name]
    print(f"\nBest tree model (by validation PR-AUC): {best_name}")

    plot_roc(y_test, test_probas, FIG_DIR / "roc_curve.png")
    plot_pr(y_test, test_probas, FIG_DIR / "pr_curve.png")
    plot_confusion(y_test, test_probas[best_name], best_name, FIG_DIR / "confusion_matrix_best.png")
    plot_shap_summary(best_model, Xte, FIG_DIR / "shap_summary.png")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump(FEATURES, f, indent=2)
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "raw_features": RAW_FEATURES,
                "engineered_features": ENGINEERED_FEATURES,
                "seed": SEED,
                "test_metrics": results[results["Model"] == best_name].iloc[0].to_dict(),
            },
            f, indent=2, default=float,
        )

    print(f"\nSaved model artifacts to {MODELS_DIR}")
    print(f"Saved benchmark table and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()