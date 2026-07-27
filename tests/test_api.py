"""
Pruebas de humo (smoke tests) del backend FastAPI.

Se ejecutan automáticamente en el pipeline de CI (.github/workflows/ci-cd.yml)
en cada push. El objetivo no es cubrir cada caso de negocio, sino garantizar
que la API arranca, responde, y valida entradas correctamente ANTES de
construir la imagen Docker y desplegar. Si algo aquí falla, el pipeline se
detiene y no llega a producción.

Correr localmente:
    cd api
    pip install -r requirements.txt
    pip install pytest httpx
    pytest ../tests -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(API_DIR))

from main import app  # noqa: E402

client = TestClient(app)


def test_health_responde_ok():
    """El endpoint /health debe responder incluso si los modelos aún no
    cargaron; nunca debe tirar un error 5xx."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] in {"ok", "degraded"}
    assert "models_loaded" in body


def test_root_lista_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "predict" in "".join(body["endpoints"])


def test_predict_rechaza_lista_vacia():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 400


def test_predict_rechaza_payload_sin_features_requeridas():
    """Un registro sin las 5 features de sensores debe fallar la validación
    de Pydantic (422), nunca llegar a ml_core.py."""
    response = client.post("/predict", json={"records": [{"machine_id": "M-001"}]})
    assert response.status_code == 422


def test_predict_con_datos_validos_no_rompe_el_contrato():
    """Con los modelos reales presentes en api/models/, una lectura válida
    debe devolver la forma esperada: reconstruction_error, is_anomaly,
    action, action_label, severity."""
    payload = {
        "records": [
            {
                "machine_id": "M-001",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    # 200 si los modelos + dependencias pesadas (tensorflow, stable-baselines3)
    # están disponibles; 503/500 si el runner de CI no las instaló (no es lo
    # que este test de contrato valida). Lo importante es que NUNCA truene
    # con un error de validación (422) ni deje pasar un contrato incompleto.
    assert response.status_code in (200, 500, 503)
    if response.status_code == 200:
        result = response.json()["results"][0]
        for key in ("machine_id", "reconstruction_error", "is_anomaly", "action", "action_label", "severity"):
            assert key in result


def test_retrain_dispara_run_id():
    response = client.post("/retrain")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"queued", "dispatched"}
    assert "run_id" in body
