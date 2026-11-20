import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'api' / 'predictions.db').as_posix()}")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

IS_SQLITE = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if IS_SQLITE else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    inputs: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    risk_tier: Mapped[str] = mapped_column(String(16), index=True)
    top_shap_factors: Mapped[list] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_used: Mapped[str] = mapped_column(String(64))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(
    db: Session,
    inputs: dict,
    probability: float,
    risk_tier: str,
    top_shap_factors: list,
    latency_ms: float,
    model_used: str,
) -> PredictionLog:
    record = PredictionLog(
        inputs=inputs,
        probability=probability,
        risk_tier=risk_tier,
        top_shap_factors=top_shap_factors,
        latency_ms=latency_ms,
        model_used=model_used,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def recent_predictions(db: Session, limit: int = 20) -> list:
    stmt = select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)
    return list(db.execute(stmt).scalars())