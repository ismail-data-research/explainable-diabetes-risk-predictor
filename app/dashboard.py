import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Explainable Early Disease Risk Predictor", page_icon="🩺", layout="wide")

FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check (5 yrs)",
    "BMI": "Body mass index",
    "Smoker": "Smoking history",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease / heart attack",
    "PhysActivity": "Physical activity",
    "Fruits": "Daily fruit intake",
    "Veggies": "Daily vegetable intake",
    "HvyAlcoholConsump": "Heavy alcohol use",
    "AnyHealthcare": "Health coverage",
    "NoDocbcCost": "Skipped doctor due to cost",
    "GenHlth": "General health rating",
    "MentHlth": "Poor mental health days",
    "PhysHlth": "Poor physical health days",
    "DiffWalk": "Difficulty walking",
    "Sex": "Sex",
    "Age": "Age group",
    "Education": "Education level",
    "Income": "Income level",
    "Health_Risk_Score": "Clinical risk score",
    "BMI_HighBP_Interaction": "BMI x high blood pressure",
    "Lifestyle_Risk_Score": "Lifestyle risk score",
}

BINARY_FEATURES = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk",
}

GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
AGE = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
    8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80 or older",
}
EDUCATION = {
    1: "Never attended school / kindergarten only", 2: "Elementary (grades 1-8)",
    3: "Some high school", 4: "High school graduate",
    5: "Some college or technical school", 6: "College graduate",
}
INCOME = {
    1: "Less than $10,000", 2: "$10,000 to $15,000", 3: "$15,000 to $20,000",
    4: "$20,000 to $25,000", 5: "$25,000 to $35,000", 6: "$35,000 to $50,000",
    7: "$50,000 to $75,000", 8: "$75,000 or more",
}

TIER_COLORS = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}
TIER_GUIDANCE = {
    "Low": "Your profile shows a lower screening risk. Keep up healthy habits and follow routine screening guidance from your clinician.",
    "Medium": "Your profile shows a moderate screening risk. Consider discussing a blood glucose or HbA1c test with a healthcare professional.",
    "High": "Your profile shows an elevated screening risk. Please arrange a visit with a healthcare professional for confirmatory testing such as fasting glucose or HbA1c.",
}
FACTOR_TIPS = {
    "BMI": "Gradual, sustainable weight management through diet and activity can lower risk.",
    "BMI_HighBP_Interaction": "Weight management and blood pressure control work together to reduce risk.",
    "HighBP": "Ask your clinician about blood pressure monitoring and management.",
    "HighChol": "Ask your clinician about cholesterol testing, diet, and treatment options.",
    "Health_Risk_Score": "Several cardiometabolic risk factors are present; a clinical review is worthwhile.",
    "Lifestyle_Risk_Score": "Small, consistent improvements in activity, diet, and alcohol intake add up.",
    "PhysActivity": "Aim for about 150 minutes of moderate activity per week, as your health allows.",
    "Smoker": "Smoking cessation support can reduce cardiometabolic risk.",
    "Fruits": "Include more fruit in your daily diet.",
    "Veggies": "Include more vegetables in your daily diet.",
    "HvyAlcoholConsump": "Reducing alcohol intake supports healthier blood sugar control.",
    "GenHlth": "A routine checkup can help address concerns behind your general health rating.",
    "PhysHlth": "Persistent physical health issues deserve a clinical review.",
    "MentHlth": "Mental health and metabolic health are linked; consider seeking support if needed.",
    "DiffWalk": "Ask about safe, low-impact activity options that suit your mobility.",
    "HeartDiseaseorAttack": "Keep up regular cardiology follow-up and medication adherence.",
    "Stroke": "Keep up regular follow-up with your care team.",
    "Age": "Risk rises with age, so regular screening is especially useful.",
    "NoDocbcCost": "Look into community clinics or low-cost screening programs.",
    "AnyHealthcare": "Explore available coverage or community health programs.",
    "CholCheck": "Schedule a cholesterol check if you have not had one recently.",
    "Income": "Community health programs may offer free or low-cost screening.",
    "Education": "Public health resources can help with accessible health information.",
}


def format_value(feature: str, value: float) -> str:
    if feature in BINARY_FEATURES:
        return "Yes" if value >= 0.5 else "No"
    if feature == "Sex":
        return "Male" if value >= 0.5 else "Female"
    if feature == "Age":
        return AGE.get(int(value), str(int(value)))
    if feature == "GenHlth":
        return GENHLTH.get(int(value), str(int(value)))
    return f"{value:g}"


def yes_no(label: str, default: int = 0) -> int:
    return 1 if st.radio(label, ["No", "Yes"], index=default, horizontal=True) == "Yes" else 0


def call_api(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def risk_gauge(probability: float, tier: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": f"Risk Tier: {tier}", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": TIER_COLORS[tier], "thickness": 0.3},
                "steps": [
                    {"range": [0, 35], "color": "#d5f0df"},
                    {"range": [35, 70], "color": "#fdecc4"},
                    {"range": [70, 100], "color": "#f6d0d0"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": probability * 100},
            },
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def shap_chart(factors: list) -> go.Figure:
    ordered = sorted(factors, key=lambda f: f["shap_value"])
    labels = [f"{FEATURE_LABELS.get(f['feature'], f['feature'])} = {format_value(f['feature'], f['value'])}" for f in ordered]
    values = [f["shap_value"] for f in ordered]
    colors = ["#d64545" if v > 0 else "#2e9e5b" for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in values], textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    span = max(abs(v) for v in values) * 1.35 if values else 1
    fig.update_layout(
        height=380, margin=dict(l=10, r=30, t=30, b=40),
        xaxis=dict(title="Contribution to risk score (red raises risk, green lowers risk)", range=[-span, span]),
        yaxis=dict(automargin=True),
    )
    return fig


def build_payload() -> dict:
    payload = {}
    st.sidebar.title("Patient Health Indicators")
    with st.sidebar.form("patient_form"):
        st.subheader("Demographics")
        payload["Sex"] = 1 if st.radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0
        payload["Age"] = st.selectbox("Age group", list(AGE), index=7, format_func=AGE.get)
        payload["Education"] = st.selectbox("Education level", list(EDUCATION), index=3, format_func=EDUCATION.get)
        payload["Income"] = st.selectbox("Annual household income", list(INCOME), index=5, format_func=INCOME.get)

        st.subheader("Body & Clinical")
        payload["BMI"] = st.number_input("Body mass index (BMI)", min_value=12.0, max_value=98.0, value=27.0, step=0.5)
        payload["HighBP"] = yes_no("High blood pressure")
        payload["HighChol"] = yes_no("High cholesterol")
        payload["CholCheck"] = yes_no("Cholesterol checked in last 5 years", 1)
        payload["Stroke"] = yes_no("Ever had a stroke")
        payload["HeartDiseaseorAttack"] = yes_no("Heart disease or heart attack")
        payload["DiffWalk"] = yes_no("Difficulty walking or climbing stairs")

        st.subheader("Lifestyle")
        payload["Smoker"] = yes_no("Smoked 100+ cigarettes in lifetime")
        payload["PhysActivity"] = yes_no("Physical activity in past 30 days", 1)
        payload["Fruits"] = yes_no("Eats fruit daily", 1)
        payload["Veggies"] = yes_no("Eats vegetables daily", 1)
        payload["HvyAlcoholConsump"] = yes_no("Heavy alcohol consumption")

        st.subheader("Health Status & Access")
        payload["GenHlth"] = st.select_slider("General health", options=list(GENHLTH), value=3, format_func=GENHLTH.get)
        payload["MentHlth"] = st.slider("Poor mental health days (past 30)", 0, 30, 0)
        payload["PhysHlth"] = st.slider("Poor physical health days (past 30)", 0, 30, 0)
        payload["AnyHealthcare"] = yes_no("Has health coverage", 1)
        payload["NoDocbcCost"] = yes_no("Skipped doctor visit due to cost")

        submitted = st.form_submit_button("Assess Risk", use_container_width=True, type="primary")
    return payload, submitted


def render_results(result: dict) -> None:
    probability, tier = result["probability"], result["risk_tier"]
    left, right = st.columns([1, 1.4])
    with left:
        st.plotly_chart(risk_gauge(probability, tier), use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;'>Estimated screening probability: "
            f"<b style='color:{TIER_COLORS[tier]}'>{probability:.1%}</b></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Why this score?")
        st.plotly_chart(shap_chart(result["top_shap_factors"]), use_container_width=True)

    st.subheader("Preventative Guidance")
    box = {"Low": st.success, "Medium": st.warning, "High": st.error}[tier]
    box(TIER_GUIDANCE[tier])
    risk_raising = [f for f in result["top_shap_factors"] if f["shap_value"] > 0]
    if risk_raising:
        st.markdown("**Factors raising your score and what you can do:**")
        for f in risk_raising:
            name = FEATURE_LABELS.get(f["feature"], f["feature"])
            st.markdown(f"- **{name}:** {FACTOR_TIPS.get(f['feature'], 'Discuss this factor with a healthcare professional.')}")
    else:
        st.markdown("None of the top contributing factors are raising your score. Keep up the healthy habits.")

    st.caption(
        f"Model: {result['model_used']} | Inference latency: {result['latency_ms']:.1f} ms | "
        f"Log ID: {result.get('log_id')}"
    )


def main() -> None:
    st.title("🩺 Explainable Early Disease Risk Predictor")
    st.caption("Diabetes / prediabetes screening support powered by gradient-boosted trees and SHAP explanations.")

    payload, submitted = build_payload()

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.sidebar.success(f"API online ({health.get('model', 'model')})")
    except requests.RequestException:
        st.sidebar.error(f"API unreachable at {API_URL}")

    if submitted:
        try:
            with st.spinner("Computing risk and explanation..."):
                result = call_api(payload)
            render_results(result)
        except requests.HTTPError as exc:
            st.error(f"API returned an error: {exc.response.status_code} {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API at {API_URL}. Start it first. Details: {exc}")
    else:
        st.info("Fill in the health indicators in the sidebar and press **Assess Risk**.")

    st.markdown("---")
    st.warning(
        "**Medical disclaimer:** This tool is a screening support aid, not a clinical diagnosis. "
        "It estimates statistical risk from population survey data (CDC BRFSS 2015) and cannot replace "
        "evaluation by a qualified healthcare professional. Consult a clinician for diagnosis and treatment decisions."
    )


main()
