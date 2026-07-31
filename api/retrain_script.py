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
def run_training(data_path: Path, output_dir: Path, dry_run: bool = False):
    """
    Re-entrena Autoencoder + DQN y guarda artefactos en output_dir.
    En dry_run=True: usa models_new/ (carpeta temporal).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Carga de datos ------------------------------------------------
    import pandas as pd

    print(f"[TRAIN] Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)

    SENSOR_COLS = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        _fail(f"Columnas faltantes en el CSV: {missing}")

    if len(df) < GATE_MIN_ROWS:
        _fail(
            f"Dataset insuficiente: {len(df)} filas < mínimo {GATE_MIN_ROWS}. "
            "El model gate rechazará este entrenamiento."
        )

    print(f"[TRAIN] Dataset: {len(df)} filas × {len(SENSOR_COLS)} sensores")

    # ---- Preprocesado --------------------------------------------------
    from sklearn.preprocessing import StandardScaler

    # Entrenamos el Autoencoder SOLO con muestras sin falla (etiqueta 0)
    if "Machine failure" in df.columns:
        df_normal = df[df["Machine failure"] == 0].copy()
        df_fail = df[df["Machine failure"] == 1].copy()
    else:
        # Sin etiqueta: usar todo para entrenar
        df_normal = df.copy()
        df_fail = df.sample(frac=0.05, random_state=42)

    X_normal = df_normal[SENSOR_COLS].values.astype("float32")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    # ---- Autoencoder ---------------------------------------------------
    print("[TRAIN] Entrenando Autoencoder...")

    # Configuración de TensorFlow ANTES de importarlo para prevenir
    # Segmentation Fault en runners con memoria limitada (ej. GitHub Actions ~7GB).
    import os as _os
    _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # desactiva oneDNN (ahorra RAM)
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # suprime logs verbosos
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # fuerza CPU, evita init GPU

    import tensorflow as tf

    # Limitar crecimiento de memoria: evita que TF reserve toda la RAM disponible.
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    n_features = len(SENSOR_COLS)
    inp = tf.keras.Input(shape=(n_features,))
    enc = tf.keras.layers.Dense(8, activation="relu")(inp)
    bot = tf.keras.layers.Dense(4, activation="relu")(enc)
    dec = tf.keras.layers.Dense(8, activation="relu")(bot)
    out = tf.keras.layers.Dense(n_features, activation="linear")(dec)
    ae = tf.keras.Model(inputs=inp, outputs=out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
    print(f"[TRAIN] Autoencoder entrenado. Parámetros: {ae.count_params()}")

    # ---- Umbral (percentil 95 sobre datos normales) --------------------
    X_test_normal = scaler.transform(df_normal[SENSOR_COLS].values.astype("float32"))
    recon_normal = ae.predict(X_test_normal, verbose=0)
    mse_normal = np.mean((X_test_normal - recon_normal) ** 2, axis=1)
    threshold = float(np.percentile(mse_normal, 95))
    print(f"[TRAIN] Umbral calculado (p95 normal): {threshold:.6f}")

    # ---- Métricas sobre datos de falla ---------------------------------
    if len(df_fail) > 0:
        X_fail = scaler.transform(df_fail[SENSOR_COLS].values.astype("float32"))
        recon_fail = ae.predict(X_fail, verbose=0)
        mse_fail = np.mean((X_fail - recon_fail) ** 2, axis=1)
        pct_detected = float(np.mean(mse_fail > threshold) * 100)
    else:
        mse_fail = np.array([])
        pct_detected = 0.0

    metrics = {
        "reconstruction_mse_normal_mean": float(np.mean(mse_normal)),
        "reconstruction_mse_falla_mean": float(np.mean(mse_fail)) if len(mse_fail) > 0 else None,
        "pct_fallas_sobre_umbral": pct_detected,
        "n_train_rows": len(df_normal),
        "n_total_rows": len(df),
    }
    print(f"[TRAIN] Métricas: {metrics}")

    # ---- DQN (reentrenamiento rápido con nuevo threshold) --------------
    print("[TRAIN] Reentrenando agente DQN...")
    try:
        _retrain_dqn(threshold, output_dir)
    except Exception as e:
        print(f"[TRAIN] Advertencia DQN: {e}. Se reutilizará el DQN actual.")
        # Copiar DQN existente si el reentrenamiento falla
        src_dqn = MODELS_DIR / "dqn_maintenance_agent.zip"
        if src_dqn.exists():
            shutil.copy2(src_dqn, output_dir / "dqn_maintenance_agent.zip")

    # ---- Guardar artefactos --------------------------------------------
    import joblib

    ae.save(str(output_dir / "autoencoder.h5"))
    joblib.dump(scaler, output_dir / "scaler.pkl")

    config = {
        "sensor_cols": SENSOR_COLS,
        "error_threshold": threshold,
        "costos": {"costo_falla": 100, "costo_parada_preventiva": 5, "costo_parada_innecesaria": 10},
        "normalizacion": {"tool_wear_max": 253.0, "torque_max": 76.6},
        "acciones": {"0": "operar", "1": "mantenimiento_preventivo", "2": "parada_emergencia"},
        "observation_space": [
            "reconstruction_error", "tiempo_desde_mantenimiento_norm",
            "tool_wear_norm", "torque_norm"
        ],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Guardar métricas intermedias para el gate
    (output_dir / "train_metrics.json").write_text(
        json.dumps({"metrics": metrics, "threshold": threshold, "trained_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )

    print(f"[TRAIN] Artefactos guardados en: {output_dir}")
    return metrics, threshold


def _retrain_dqn(threshold: float, output_dir: Path):
    """Entrena brevemente el agente DQN con el nuevo threshold."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import DQN

    class MaintenanceEnv(gym.Env):
        """Entorno simplificado para demo de reentrenamiento DQN."""
        observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(3)

        def reset(self, seed=None, options=None):
            self.step_count = 0
            self.state = np.random.rand(4).astype(np.float32)
            return self.state, {}

        def step(self, action):
            self.step_count += 1
            error = float(self.state[0])
            is_anomaly = error > threshold

            if is_anomaly and action == 2:
                reward = 10.0
            elif is_anomaly and action == 1:
                reward = 5.0
            elif not is_anomaly and action == 0:
                reward = 1.0
            else:
                reward = -5.0

            self.state = np.random.rand(4).astype(np.float32)
            done = self.step_count >= 100
            return self.state, reward, done, False, {}

    env = MaintenanceEnv()
    model = DQN("MlpPolicy", env, learning_rate=1e-3, verbose=0)
    model.learn(total_timesteps=5000)
    model.save(str(output_dir / "dqn_maintenance_agent"))
    print("[TRAIN] Agente DQN reentrenado y guardado.")


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
