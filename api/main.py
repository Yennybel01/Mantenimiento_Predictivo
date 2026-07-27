"""
main.py — Capa de Backend (FastAPI).

Define los endpoints REST que consume el dashboard de Streamlit (y que
podría consumir cualquier otro cliente: una app móvil, un ERP, etc.).
Este archivo NO conoce detalles de UI; solo orquesta llamadas a ml_core.py
y valida entradas/salidas con Pydantic.

Correr localmente:
    uvicorn main:app --reload --port 8000

Documentación interactiva autogenerada:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ml_core


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precalentar modelos al iniciar la API
    ml_core.warmup()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Autoencoder (detección de anomalías) + Agente DQN (decisión de mantenimiento)",
    version=ml_core.load_metadata().get("version", "1.0.0"),
    lifespan=lifespan,
)

# En producción, restringir allow_origins al dominio real del dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Esquemas (contrato de la API)
# --------------------------------------------------------------------------
class SensorReading(BaseModel):
    machine_id: str = "M-001"
    air_temperature_k: float = Field(..., alias="Air temperature [K]")
    process_temperature_k: float = Field(..., alias="Process temperature [K]")
    rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]")
    torque_nm: float = Field(..., alias="Torque [Nm]")
    tool_wear_min: float = Field(..., alias="Tool wear [min]")
    tiempo_desde_mantenimiento_min: float = 0.0

    class Config:
        populate_by_name = True


class PredictRequest(BaseModel):
    records: List[SensorReading]


class PredictResult(BaseModel):
    machine_id: str
    reconstruction_error: float
    is_anomaly: bool
    action: int
    action_label: str
    severity: str


class PredictResponse(BaseModel):
    results: List[PredictResult]


class RetrainResponse(BaseModel):
    run_id: str
    status: str
    triggered_at: str


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if ml_core.models_available() else "degraded",
        "models_loaded": ml_core.models_available(),
        "uptime_seconds": ml_core.uptime_seconds(),
    }


@app.get("/model/metadata")
def model_metadata():
    metadata = ml_core.load_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="model_metadata.json no encontrado")
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="La lista 'records' no puede estar vacía")
    try:
        records = [r.dict(by_alias=True) for r in payload.records]
        results = ml_core.predict(records)
        return {"results": results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")


@app.get("/alerts")
def alerts(limit: int = 50, severity: Optional[str] = None):
    return {"alerts": ml_core.get_alerts(limit=limit, severity=severity)}


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    """Placeholder: en producción esto dispararía el pipeline de CI/CD
    (ver .github/workflows/retrain.yml) que reentrena el modelo con datos
    nuevos y publica una nueva versión en el registro de modelos."""
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return {
        "run_id": run_id,
        "status": "queued",
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    return {
        "service": "predictive-maintenance-api",
        "docs": "/docs",
        "endpoints": ["/health", "/model/metadata", "/predict", "/alerts", "/retrain"],
    }
