import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.database import get_db, init_db, log_prediction, recent_predictions
from src.explainability import RiskExplainer
from src.train_and_evaluate import MODELS_DIR, RAW_FEATURES, engineer_features

logger = logging.getLogger("risk_api")
logging.basicConfig(level=logging.INFO)

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.70

Binary = Annotated[int, Field(ge=0, le=1)]


class PatientInput(BaseModel):
    HighBP: Binary = Field(..., description="High blood pressure (0/1)")
    HighChol: Binary = Field(..., description="High cholesterol (0/1)")
    CholCheck: Binary = Field(..., description="Cholesterol check in last 5 years (0/1)")
    BMI: float = Field(..., ge=12, le=98, description="Body mass index")
    Smoker: Binary = Field(..., description="Smoked at least 100 cigarettes in lifetime (0/1)")
    Stroke: Binary = Field(..., description="Ever had a stroke (0/1)")
    HeartDiseaseorAttack: Binary = Field(..., description="Coronary heart disease or myocardial infarction (0/1)")
    PhysActivity: Binary = Field(..., description="Physical activity in past 30 days (0/1)")
    Fruits: Binary = Field(..., description="Consumes fruit daily (0/1)")
    Veggies: Binary = Field(..., description="Consumes vegetables daily (0/1)")
    HvyAlcoholConsump: Binary = Field(..., description="Heavy alcohol consumption (0/1)")
    AnyHealthcare: Binary = Field(..., description="Has any health care coverage (0/1)")
    NoDocbcCost: Binary = Field(..., description="Could not see doctor because of cost (0/1)")
    GenHlth: int = Field(..., ge=1, le=5, description="General health, 1=Excellent to 5=Poor")
    MentHlth: int = Field(..., ge=0, le=30, description="Days of poor mental health in past 30 days")
    PhysHlth: int = Field(..., ge=0, le=30, description="Days of poor physical health in past 30 days")
    DiffWalk: Binary = Field(..., description="Difficulty walking or climbing stairs (0/1)")
    Sex: Binary = Field(..., description="0=Female, 1=Male")
    Age: int = Field(..., ge=1, le=13, description="Age category, 1=18-24 to 13=80+")
    Education: int = Field(..., ge=1, le=6, description="Education level, 1 to 6")
    Income: int = Field(..., ge=1, le=8, description="Income category, 1 to 8")


class ShapFactor(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    probability: float
    risk_tier: str
    top_shap_factors: List[ShapFactor]
    latency_ms: float
    model_used: str
    log_id: int | None = None


def risk_tier(probability: float) -> str:
    if probability < LOW_THRESHOLD:
        return "Low"
    if probability <= HIGH_THRESHOLD:
        return "Medium"
    return "High"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    app.state.explainer = RiskExplainer()
    app.state.model_used = metadata["best_model"]
    logger.info("Loaded model: %s", app.state.model_used)
    yield


app = FastAPI(
    title="Explainable Early Disease Risk Predictor API",
    version="1.0.0",
    description="Screening-support API returning diabetes risk probability with SHAP explanations.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.model_used}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput, request: Request, db: Session = Depends(get_db)):
    svc: RiskExplainer = request.app.state.explainer
    inputs = payload.model_dump()

    start = time.perf_counter()
    raw = pd.DataFrame([inputs])[RAW_FEATURES].astype(float)
    X = engineer_features(raw)[svc.features]
    probability = float(svc.model.predict_proba(X)[0, 1])
    factors = svc.top_factors(X, k=5)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    tier = risk_tier(probability)

    log_id = None
    try:
        record = log_prediction(
            db, inputs, probability, tier, factors, latency_ms, request.app.state.model_used
        )
        log_id = record.id
    except Exception as exc:
        db.rollback()
        logger.error("Failed to log prediction: %s", exc)

    return PredictionResponse(
        probability=round(probability, 4),
        risk_tier=tier,
        top_shap_factors=factors,
        latency_ms=latency_ms,
        model_used=request.app.state.model_used,
        log_id=log_id,
    )


@app.get("/predictions/recent")
def recent(limit: int = 20, db: Session = Depends(get_db)):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "probability": r.probability,
            "risk_tier": r.risk_tier,
            "latency_ms": r.latency_ms,
            "model_used": r.model_used,
        }
        for r in recent_predictions(db, limit)
    ]