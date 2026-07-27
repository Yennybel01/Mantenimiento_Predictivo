"""
ml_core.py — capa de acceso a los modelos de mantenimiento predictivo.

Responsabilidad única de este módulo: dado un conjunto de lecturas de
sensores, devolver el error de reconstrucción del Autoencoder y la acción
recomendada por el agente DQN. Es la ÚNICA parte del sistema que conoce los
artefactos de ML (autoencoder.h5, scaler.pkl, config.json,
dqn_maintenance_agent.zip). Ni el frontend ni ningún otro módulo deberían
cargar estos archivos directamente — todos pasan por aquí, vía la API.

Mantenimiento: si el equipo de train entrega una nueva versión del modelo,
basta con reemplazar los archivos en models/ y actualizar model_metadata.json.
No hay que tocar main.py.
"""

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
AUTOENCODER_SCRIPT = BASE_DIR / "autoencoder_infer.py"
ALERTS_LOG_PATH = BASE_DIR / "alerts_log.jsonl"

ACCIONES_LABELS = {
    0: "Operar con normalidad",
    1: "Mantenimiento preventivo",
    2: "Parada de emergencia",
}
ACCIONES_SEVERITY = {0: "ok", 1: "warning", 2: "critical"}

_START_TIME = time.time()

# Subproceso persistente del autoencoder: se arranca UNA vez (en warmup) y
# se reutiliza en cada predicción, para no pagar el costo de reimportar
# TensorFlow por cada request (ver autoencoder_infer.py).
_AE_PROC: Optional[subprocess.Popen] = None
_AE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Metadata / config
# --------------------------------------------------------------------------
def load_metadata() -> dict:
    path = MODELS_DIR / "model_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_config() -> dict:
    path = MODELS_DIR / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def uptime_seconds() -> float:
    return round(time.time() - _START_TIME, 1)


def models_available() -> bool:
    required = ["autoencoder.h5", "scaler.pkl", "config.json", "dqn_maintenance_agent.zip"]
    return all((MODELS_DIR / f).exists() for f in required)


# --------------------------------------------------------------------------
# Autoencoder (subproceso) + agente DQN (en proceso)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_dqn_agent():
    # pyrefly: ignore [missing-import]
    from stable_baselines3 import DQN

    return DQN.load(MODELS_DIR / "dqn_maintenance_agent.zip")


def _start_autoencoder_proc() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(AUTOENCODER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )
    # Espera la línea {"ready": true} antes de aceptar predicciones.
    ready_line = proc.stdout.readline()
    if not ready_line or "ready" not in ready_line:
        stderr_tail = proc.stderr.read()[-2000:] if proc.stderr else ""
        raise RuntimeError(f"El subproceso del autoencoder no arrancó correctamente: {stderr_tail}")
    return proc


def _get_autoencoder_proc() -> subprocess.Popen:
    global _AE_PROC
    if _AE_PROC is None or _AE_PROC.poll() is not None:
        # None = nunca arrancó; poll() != None = se cayó, hay que reiniciarlo.
        _AE_PROC = _start_autoencoder_proc()
    return _AE_PROC


def _reconstruction_errors(rows: list, sensor_cols: list) -> np.ndarray:
    payload = {"sensor_cols": sensor_cols, "rows": rows}
    with _AE_LOCK:
        proc = _get_autoencoder_proc()
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            out_line = proc.stdout.readline()
        except (BrokenPipeError, OSError):
            # El subproceso murió (ej. OOM); lo reiniciamos y reintentamos una vez.
            global _AE_PROC
            _AE_PROC = None
            proc = _get_autoencoder_proc()
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            out_line = proc.stdout.readline()

        if not out_line:
            stderr_tail = proc.stderr.read()[-2000:] if proc.stderr else ""
            raise RuntimeError(f"El subproceso del autoencoder no respondió: {stderr_tail}")

        result = json.loads(out_line)
        if "error" in result:
            raise RuntimeError(f"Error en el subproceso del autoencoder: {result['error']}")
        return np.array(result["errors"])


def _build_observations(records: list, errors: np.ndarray, config: dict) -> np.ndarray:
    norm = config.get("normalizacion", {"tool_wear_max": 253.0, "torque_max": 76.6})
    tool_wear_max = norm.get("tool_wear_max", 253.0)
    torque_max = norm.get("torque_max", 76.6)

    obs = []
    for rec, err in zip(records, errors):
        tool_wear = rec.get("Tool wear [min]", 0.0)
        torque = rec.get("Torque [Nm]", 0.0)
        tiempo = rec.get("tiempo_desde_mantenimiento_min", 0.0)
        tiempo_norm = min(tiempo / 200.0, 1.0)
        obs.append([err, tiempo_norm, tool_wear / tool_wear_max, torque / torque_max])
    return np.array(obs, dtype=np.float32)


def _log_alert(record: dict, error: float, action: int, machine_id: str):
    if action == 0:
        return  # solo se registran acciones que requieren atención
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": machine_id,
        "reconstruction_error": round(float(error), 5),
        "action": action,
        "action_label": ACCIONES_LABELS[action],
        "severity": ACCIONES_SEVERITY[action],
        "reading": record,
    }
    with open(ALERTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_alerts(limit: int = 50, severity: Optional[str] = None) -> list:
    if not ALERTS_LOG_PATH.exists():
        return []
    lines = ALERTS_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    if severity:
        entries = [e for e in entries if e["severity"] == severity]
    return list(reversed(entries))[:limit]


# --------------------------------------------------------------------------
# Punto de entrada principal
# --------------------------------------------------------------------------
def predict(records: list, log_alerts: bool = True) -> list:
    """records: lista de dicts con las 5 features de sensores (+ opcional
    machine_id y tiempo_desde_mantenimiento_min). Devuelve una lista de
    resultados con reconstruction_error, is_anomaly, action, action_label."""
    if not models_available():
        raise FileNotFoundError(
            "Faltan artefactos del modelo en api/models/. Se esperan: "
            "autoencoder.h5, scaler.pkl, config.json, dqn_maintenance_agent.zip"
        )

    config = load_config()
    sensor_cols = config.get("sensor_cols")
    threshold = config.get("error_threshold", 0.35)

    rows = [[rec[c] for c in sensor_cols] for rec in records]
    errors = _reconstruction_errors(rows, sensor_cols)
    obs = _build_observations(records, errors, config)

    agent = _load_dqn_agent()
    results = []
    for rec, err, ob in zip(records, errors, obs):
        action, _ = agent.predict(ob, deterministic=True)
        action = int(action)
        machine_id = rec.get("machine_id", "unknown")
        result = {
            "machine_id": machine_id,
            "reconstruction_error": round(float(err), 5),
            "is_anomaly": bool(err > threshold),
            "action": action,
            "action_label": ACCIONES_LABELS[action],
            "severity": ACCIONES_SEVERITY[action],
        }
        results.append(result)
        if log_alerts:
            _log_alert(rec, err, action, machine_id)

    return results


def warmup():
    """Realiza una inferencia dummy para calentar los modelos (cargar DQN en
    caché e inicializar el subproceso de TensorFlow) al arrancar la API.
    Evita el retraso (ReadTimeout) en la primera consulta real."""
    if not models_available():
        print("[Warmup] Modelos no disponibles en api/models/. Omitiendo precalentamiento.")
        return
    
    print("[Warmup] Iniciando precalentamiento de modelos...")
    
    # 1. Cargar el agente DQN (se almacena en caché a través de @lru_cache)
    try:
        start_dqn = time.time()
        _load_dqn_agent()
        print(f"[Warmup] Agente DQN cargado con éxito en {time.time() - start_dqn:.2f}s.")
    except Exception as e:
        print(f"[Warmup] Advertencia al cargar el agente DQN: {e}", file=sys.stderr)
        
    # 2. Forzar un primer arranque del subproceso TensorFlow con una predicción dummy
    try:
        start_ae = time.time()
        config = load_config()
        sensor_cols = config.get("sensor_cols")
        if sensor_cols:
            dummy_row = [0.0] * len(sensor_cols)
            _reconstruction_errors([dummy_row], sensor_cols)
            print(f"[Warmup] Autoencoder (subproceso TensorFlow) precalentado en {time.time() - start_ae:.2f}s.")
        else:
            print("[Warmup] Advertencia: 'sensor_cols' no especificado en la configuración. No se pudo calentar el Autoencoder.")
    except Exception as e:
        print(f"[Warmup] Advertencia al precalentar el Autoencoder: {e}", file=sys.stderr)