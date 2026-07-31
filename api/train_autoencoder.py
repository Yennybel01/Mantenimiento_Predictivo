"""
train_autoencoder.py — Entrenamiento del Autoencoder en proceso AISLADO.

Por qué separado: TensorFlow y PyTorch/stable-baselines3 chocan a nivel de
librerías nativas (BLAS, oneDNN, thread pools) cuando coexisten en el mismo
proceso → Segmentation Fault (exit 139). La misma razón por la que
autoencoder_infer.py corre como subproceso en ml_core.py.

Uso (siempre vía retrain_script.py, nunca directamente):
    python api/train_autoencoder.py \\
        --data-path data/ai4i2020.csv \\
        --output-dir api/models_new

Salida:
    Escribe en output-dir: autoencoder.h5, scaler.pkl, config.json,
    train_metrics.json y sale con código 0 si tuvo éxito, 1 si falló.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
GATE_MIN_ROWS = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Carga de datos ──────────────────────────────────────────────────────
    import pandas as pd

    print(f"[AE-TRAIN] Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)

    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        print(f"[AE-TRAIN] ERROR: Columnas faltantes: {missing}", file=sys.stderr)
        sys.exit(1)

    if len(df) < GATE_MIN_ROWS:
        print(f"[AE-TRAIN] ERROR: Dataset insuficiente ({len(df)} filas)", file=sys.stderr)
        sys.exit(1)

    print(f"[AE-TRAIN] Dataset: {len(df)} filas × {len(SENSOR_COLS)} sensores")

    # ── Preprocesado ────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler

    if "Machine failure" in df.columns:
        df_normal = df[df["Machine failure"] == 0].copy()
        df_fail = df[df["Machine failure"] == 1].copy()
    else:
        df_normal = df.copy()
        df_fail = df.sample(frac=0.05, random_state=42)

    X_normal = df_normal[SENSOR_COLS].values.astype("float32")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    # ── Autoencoder (TensorFlow — único import en este proceso) ─────────────
    print("[AE-TRAIN] Entrenando Autoencoder...")
    import tensorflow as tf

    n_features = len(SENSOR_COLS)
    inp = tf.keras.Input(shape=(n_features,))
    enc = tf.keras.layers.Dense(8, activation="relu")(inp)
    bot = tf.keras.layers.Dense(4, activation="relu")(enc)
    dec = tf.keras.layers.Dense(8, activation="relu")(bot)
    out = tf.keras.layers.Dense(n_features, activation="linear")(dec)
    ae = tf.keras.Model(inputs=inp, outputs=out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
    print(f"[AE-TRAIN] Autoencoder entrenado. Parámetros: {ae.count_params()}")

    # ── Umbral (percentil 95 sobre datos normales) ──────────────────────────
    X_test_normal = scaler.transform(df_normal[SENSOR_COLS].values.astype("float32"))
    recon_normal = ae.predict(X_test_normal, verbose=0)
    mse_normal = np.mean((X_test_normal - recon_normal) ** 2, axis=1)
    threshold = float(np.percentile(mse_normal, 95))
    print(f"[AE-TRAIN] Umbral calculado (p95 normal): {threshold:.6f}")

    # ── Métricas sobre datos de falla ───────────────────────────────────────
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
    print(f"[AE-TRAIN] Métricas: {metrics}")

    # ── Guardar artefactos ──────────────────────────────────────────────────
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
    (output_dir / "train_metrics.json").write_text(
        json.dumps({
            "metrics": metrics,
            "threshold": threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }),
        encoding="utf-8",
    )

    print(f"[AE-TRAIN] Artefactos guardados en: {output_dir}")


if __name__ == "__main__":
    main()
