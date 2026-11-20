# Explainable Early Disease Risk Predictor (Diabetes Screening)

An offline-capable, end-to-end diabetes risk screening application built on the **CDC BRFSS 2015** health indicators dataset (253,680 survey records). The system combines leakage-free **SMOTENC** preprocessing, **XGBoost**, real-time **TreeSHAP** explanations, a **FastAPI** backend with **SQLite** audit logging, and an interactive **Streamlit** dashboard.

## Key Highlights
- **Leakage-Free Preprocessing:** Exact duplicates are removed prior to a stratified 70/15/15 train-validation-test split. SMOTENC is applied strictly to the training set so binary clinical flags remain valid and test metrics stay honest.
- **Feature Engineering:** Adds three domain-informed composite features (`BMI_HighBP_Interaction`, `Health_Risk_Score`, and `Lifestyle_Risk_Score`).
- **Explainable AI (TreeSHAP):** Global beeswarm/bar importance and local patient-level waterfall explanations. The engineered feature **`BMI_HighBP_Interaction`** ranked as the **#1 global risk driver** (mean |SHAP| = `0.5191`), outperforming raw BMI and general health alone.
- **Full-Stack Local Deployment:** Decoupled FastAPI inference service (`POST /predict`, `GET /health`, `GET /predictions/recent`), SQLite prediction/latency logging, and a Streamlit + Plotly frontend.

## Model Benchmark (Held-Out Test Set: 34,422 Records)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7180 | 0.3139 | 0.7115 | 0.4356 | 0.7890 | 0.3894 | 0.46 |
| Random Forest | 0.7390 | 0.3279 | 0.6731 | 0.4410 | 0.7963 | 0.3981 | 16.56 |
| **XGBoost** | **0.7180** | **0.3166** | **0.7278** | **0.4412** | **0.8004** | **0.4101** | **1.90** |
| LightGBM | 0.7197 | 0.3172 | 0.7221 | 0.4407 | 0.7987 | 0.4088 | 1.32 |

## Repository Structure
```text
├── api/          # FastAPI backend (main.py) & SQLite audit logger (database.py)
├── app/          # Streamlit + Plotly interactive dashboard (dashboard.py)
├── models/       # Trained XGBoost model, scaler, feature list, and metadata
├── outputs/      # Benchmark CSVs, SHAP importance CSV, and 300-DPI figures
├── src/          # Training/evaluation pipeline & TreeSHAP explainability module
└── requirements.txt
