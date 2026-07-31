"""
retrain_script.py — Pipeline de reentrenamiento automatizado.

Modos de ejecución (argumentos):
    --backup          Crea backup del modelo actual antes de entrenar
    --dry-run         Entrena en carpeta temporal, NO toca api/models/
    --validate-only   Solo compara métricas nuevo vs viejo (model gate)
    --promote         Copia modelos validados de models_new/ a api/models/
    --rollback        Restaura la última versión en backup/
    (sin args)        Ciclo completo: backup → train → validate → promote

Uso desde GitHub Actions:
    python api/retrain_script.py --backup
    python api/retrain_script.py --data-path data/ai4i2020.csv
    python api/retrain_script.py --validate-only
    python api/retrain_script.py --promote
    python api/retrain_script.py --rollback     # solo si failure()
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
API_DIR = Path(__file__).resolve().parent
MODELS_DIR = API_DIR / "models"
MODELS_NEW_DIR = API_DIR / "models_new"
BACKUP_BASE_DIR = ROOT_DIR / "models_backup"
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATA_PATH = DATA_DIR / "ai4i2020.csv"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
GATE_REPORT_PATH = ROOT_DIR / "gate_report.json"

# -------------------------------------------------------------------
# Criterios del Model Gate
# -------------------------------------------------------------------
GATE_MIN_ROWS = 100          # Mínimo de filas para entrenar
GATE_MAX_MSE_RATIO = 1.10    # El nuevo MSE no puede ser >10% peor
GATE_MIN_DETECTION_PCT = 25  # Al menos 25% de fallas sobre el umbral


# ===================================================================
# PASO 1 — BACKUP
# ===================================================================
def backup_current_model() -> Path:
    """Copia api/models/ a models_backup/v{version}_{timestamp}/"""
    meta = _load_metadata()
    version = meta.get("version", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_BASE_DIR / f"v{version}_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for f in MODELS_DIR.iterdir():
        shutil.copy2(f, backup_dir / f.name)

    print(f"[BACKUP] Modelos respaldados en: {backup_dir}")
    # Guardar puntero al backup más reciente
    latest_path = BACKUP_BASE_DIR / "latest.txt"
    latest_path.write_text(str(backup_dir), encoding="utf-8")
    return backup_dir


# ===================================================================
# PASO 2 — ENTRENAMIENTO
# ===================================================================
AE_TRAIN_SCRIPT = API_DIR / "train_autoencoder.py"


def run_training(data_path: Path, output_dir: Path, dry_run: bool = False):
    """
    Re-entrena Autoencoder + DQN y guarda artefactos en output_dir.

    IMPORTANTE: TensorFlow (Autoencoder) y PyTorch/stable-baselines3 (DQN)
    NO pueden coexistir en el mismo proceso — sus librerías nativas (BLAS,
    oneDNN, thread pools) chocan a nivel de símbolos C++ → Segmentation Fault.
    El mismo problema que ya resolvió ml_core.py con autoencoder_infer.py.

    Solución: cada fase corre en su propio subproceso  aislado, replicando
    el patrón subprocess.Popen usado en ml_core.py → _start_autoencoder_proc().
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── FASE 1: Autoencoder (TF) en subproceso AISLADO ─────────────────────
    # train_autoencoder.py importa TensorFlow pero NUNCA PyTorch.
    # Este proceso termina antes de que el DQN arranque → cero coexistencia.
    print("[TRAIN] Lanzando entrenamiento del Autoencoder (proceso aislado TF)...")
    result_ae = subprocess.run(
        [
            sys.executable,
            str(AE_TRAIN_SCRIPT),
            "--data-path", str(data_path),
            "--output-dir", str(output_dir),
        ],
        capture_output=False,   # deja que stdout/stderr fluyan al runner
    )
    if result_ae.returncode != 0:
        _fail(
            f"El subproceso de entrenamiento del Autoencoder falló "
            f"(exit {result_ae.returncode}). Ver log arriba."
        )

    # Leer threshold y métricas escritas por el subproceso
    metrics_path = output_dir / "train_metrics.json"
    if not metrics_path.exists():
        _fail("El subproceso del Autoencoder no generó train_metrics.json")
    train_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = train_data["metrics"]
    threshold = train_data["threshold"]

    # ── FASE 2: DQN (PyTorch/SB3) en subproceso AISLADO ────────────────────
    # El proceso del Autoencoder ya terminó → TF ya no está en memoria.
    # El DQN importa PyTorch pero NUNCA TensorFlow → cero coexistencia.
    print("[TRAIN] Lanzando reentrenamiento del agente DQN (proceso aislado SB3)...")
    try:
        _retrain_dqn(threshold, output_dir)
    except Exception as e:
        print(f"[TRAIN] Advertencia DQN: {e}. Se reutilizará el DQN actual.")
        src_dqn = MODELS_DIR / "dqn_maintenance_agent.zip"
        if src_dqn.exists():
            shutil.copy2(src_dqn, output_dir / "dqn_maintenance_agent.zip")

    print(f"[TRAIN] Artefactos guardados en: {output_dir}")
    return metrics, threshold


def _retrain_dqn(threshold: float, output_dir: Path):
    """
    Entrena el agente DQN en un subproceso aislado (PyTorch/SB3).

    Se ejecuta via subprocess.run() para garantizar que TensorFlow y PyTorch
    nunca compartan el mismo espacio de proceso. El subproceso importa
    stable_baselines3 (PyTorch) pero jamas TensorFlow.
    """
    # El script DQN es inline: se genera como un archivo temporal y se ejecuta.
    # Esto evita necesitar un archivo dqn_train.py permanente en el repo.
    dqn_code = f"""
import sys, numpy as np
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import DQN

output_dir = Path(r\"{output_dir}\")
output_dir.mkdir(parents=True, exist_ok=True)
THRESHOLD = {threshold}

class MaintenanceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = np.random.rand(4).astype(np.float32)
        return self.state, {{}}
    def step(self, action):
        self.step_count += 1
        error = float(self.state[0])
        is_anomaly = error > THRESHOLD
        if is_anomaly and action == 2:   reward = 10.0
        elif is_anomaly and action == 1: reward = 5.0
        elif not is_anomaly and action == 0: reward = 1.0
        else: reward = -5.0
        self.state = np.random.rand(4).astype(np.float32)
        done = self.step_count >= 100
        return self.state, reward, done, False, {{}}

env = MaintenanceEnv()
model = DQN(\"MlpPolicy\", env, learning_rate=1e-3, verbose=0)
model.learn(total_timesteps=5000)
model.save(str(output_dir / \"dqn_maintenance_agent\"))
print(\"[DQN-TRAIN] Agente DQN reentrenado y guardado.\")
"""
    result = subprocess.run(
        [sys.executable, "-c", dqn_code],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subproceso DQN falló con exit code {result.returncode}")


# ===================================================================
# PASO 3 — VALIDACIÓN (MODEL GATE)
# ===================================================================
def validate_model(new_model_dir: Path) -> bool:
    """
    Compara el modelo nuevo (new_model_dir) con el actual (MODELS_DIR).
    Reglas del gate:
      1. MSE normal nuevo ≤ MSE viejo × 1.10
      2. % fallas detectadas ≥ GATE_MIN_DETECTION_PCT
    Retorna True si el nuevo modelo pasa, False si debe ser rechazado.
    """
    metrics_path = new_model_dir / "train_metrics.json"
    if not metrics_path.exists():
        _fail(f"No se encontró train_metrics.json en {new_model_dir}")

    new_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    new_metrics = new_data["metrics"]
    new_mse = new_metrics["reconstruction_mse_normal_mean"]
    new_pct = new_metrics["pct_fallas_sobre_umbral"]
    n_rows = new_metrics["n_train_rows"]

    old_meta = _load_metadata()
    old_mse = old_meta.get("metrics", {}).get("reconstruction_mse_normal_mean", float("inf"))

    print(f"[GATE] Filas de entrenamiento: {n_rows} (mínimo: {GATE_MIN_ROWS})")
    print(f"[GATE] MSE normal  → nuevo: {new_mse:.6f}  |  viejo: {old_mse:.6f}  |  ratio: {new_mse/old_mse:.2f}x")
    print(f"[GATE] % fallas detectadas: {new_pct:.1f}% (mínimo: {GATE_MIN_DETECTION_PCT}%)")

    gate_result = {
        "passed": False,
        "n_train_rows": n_rows,
        "new_mse": new_mse,
        "old_mse": old_mse,
        "mse_ratio": new_mse / old_mse if old_mse > 0 else None,
        "pct_detected": new_pct,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "reject_reasons": [],
    }

    reasons = []
    if n_rows < GATE_MIN_ROWS:
        reasons.append(f"Datos insuficientes: {n_rows} < {GATE_MIN_ROWS}")
    if old_mse < float("inf") and new_mse > old_mse * GATE_MAX_MSE_RATIO:
        reasons.append(f"MSE demasiado alto: {new_mse:.6f} > {old_mse * GATE_MAX_MSE_RATIO:.6f}")
    if new_pct < GATE_MIN_DETECTION_PCT:
        reasons.append(f"Detección insuficiente: {new_pct:.1f}% < {GATE_MIN_DETECTION_PCT}%")

    gate_result["reject_reasons"] = reasons
    gate_result["passed"] = len(reasons) == 0

    GATE_REPORT_PATH.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    if gate_result["passed"]:
        print("[GATE] ✅ Modelo APROBADO — listo para promoción.")
    else:
        print(f"[GATE] ❌ Modelo RECHAZADO. Razones: {reasons}")
        print(f"[GATE] Los modelos en api/models/ NO han sido modificados.")

    return gate_result["passed"]


# ===================================================================
# PASO 4 — PROMOCIÓN
# ===================================================================
def promote_model(new_model_dir: Path):
    """Copia modelos validados de new_model_dir a api/models/ y actualiza metadata."""
    gate_path = GATE_REPORT_PATH
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("passed", False):
            _fail("El model gate no fue aprobado. Usa --validate-only primero.")

    artefactos = ["autoencoder.h5", "scaler.pkl", "config.json", "dqn_maintenance_agent.zip"]
    for art in artefactos:
        src = new_model_dir / art
        if src.exists():
            shutil.copy2(src, MODELS_DIR / art)
            print(f"[PROMOTE] Copiado: {art}")

    _update_metadata(new_model_dir)
    print("[PROMOTE] ✅ Modelo promovido a producción correctamente.")


def _update_metadata(new_model_dir: Path):
    """Incrementa la versión y agrega entrada al historial de reentrenamiento."""
    meta = _load_metadata()
    metrics_data = json.loads((new_model_dir / "train_metrics.json").read_text(encoding="utf-8"))

    old_version = meta.get("version", "1.0.0")
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_version = ".".join(parts)

    meta["version"] = new_version
    meta["trained_at"] = metrics_data["trained_at"]
    meta["threshold"] = metrics_data["threshold"]
    meta["metrics"].update(metrics_data["metrics"])

    trigger = os.environ.get("RETRAIN_TRIGGER", "scheduled")
    meta.setdefault("retrain_history", []).append({
        "version": new_version,
        "trained_at": metrics_data["trained_at"],
        "trigger": trigger,
        "status": "producción",
        "metrics": metrics_data["metrics"],
    })

    METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PROMOTE] Metadata actualizada: v{old_version} → v{new_version}")


# ===================================================================
# PASO 5 — ROLLBACK
# ===================================================================
def rollback():
    """Restaura el último backup a api/models/."""
    latest_path = BACKUP_BASE_DIR / "latest.txt"
    if not latest_path.exists():
        _fail("No hay backup disponible para rollback.")

    backup_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    if not backup_dir.exists():
        _fail(f"Directorio de backup no encontrado: {backup_dir}")

    print(f"[ROLLBACK] Restaurando desde: {backup_dir}")
    for f in backup_dir.iterdir():
        shutil.copy2(f, MODELS_DIR / f.name)

    print("[ROLLBACK] ✅ Modelos anteriores restaurados correctamente.")

    # Registrar el rollback en metadata
    meta = _load_metadata()
    meta.setdefault("retrain_history", []).append({
        "version": meta.get("version", "unknown"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "rollback",
        "status": "rollback",
        "backup_source": str(backup_dir),
    })
    METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ===================================================================
# Helpers
# ===================================================================
def _load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"version": "1.0.0", "metrics": {}, "retrain_history": []}


def _fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline de reentrenamiento MLOps")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = MODELS_NEW_DIR

    if args.rollback:
        rollback()
        return

    if args.backup:
        backup_current_model()
        return

    if args.promote:
        promote_model(output_dir)
        return

    if args.validate_only:
        passed = validate_model(output_dir)
        sys.exit(0 if passed else 1)

    # Ciclo completo o dry-run
    if args.dry_run:
        # dry-run guarda en models_new/ para que los pasos --validate-only y --promote
        # del workflow encuentren los artefactos en el mismo directorio esperado.
        output_dir = MODELS_NEW_DIR
        print("[DRY-RUN] Modo dry-run: los artefactos se guardarán en models_new/, NO en api/models/")

    if not data_path.exists():
        print(f"[WARN] Dataset no encontrado en {data_path}. Generando datos sintéticos...")
        data_path = _generate_synthetic_data()

    print("\n" + "=" * 60)
    print("PIPELINE DE REENTRENAMIENTO — INICIO")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60 + "\n")

    t0 = time.time()

    if not args.dry_run:
        backup_current_model()

    metrics, threshold = run_training(data_path, output_dir)
    passed = validate_model(output_dir)

    if not args.dry_run and passed:
        promote_model(output_dir)
    elif not passed:
        print("[PIPELINE] ❌ El modelo no pasó el gate. No se promovió.")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n[PIPELINE] ✅ Completado en {elapsed:.1f}s")


def _generate_synthetic_data() -> Path:
    """Genera un CSV sintético del dataset AI4I para pruebas en CI."""
    import pandas as pd

    np.random.seed(42)
    n = 1500
    normal_n = 1400
    fail_n = 100

    normal = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300, 2, normal_n),
        "Process temperature [K]": np.random.normal(310, 1.5, normal_n),
        "Rotational speed [rpm]": np.random.normal(1500, 180, normal_n),
        "Torque [Nm]": np.random.normal(40, 10, normal_n),
        "Tool wear [min]": np.random.uniform(0, 200, normal_n),
        "Machine failure": 0,
    })
    failures = pd.DataFrame({
        "Air temperature [K]": np.random.normal(310, 5, fail_n),
        "Process temperature [K]": np.random.normal(320, 4, fail_n),
        "Rotational speed [rpm]": np.random.normal(1200, 300, fail_n),
        "Torque [Nm]": np.random.normal(65, 15, fail_n),
        "Tool wear [min]": np.random.uniform(180, 253, fail_n),
        "Machine failure": 1,
    })
    df = pd.concat([normal, failures], ignore_index=True).sample(frac=1, random_state=42)
    out = DATA_DIR / "ai4i2020_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[DATA] Dataset sintético generado: {out} ({len(df)} filas)")
    return out


if __name__ == "__main__":
    main()
