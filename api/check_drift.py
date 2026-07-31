"""
check_drift.py — Detector de drift para el monitor de GitHub Actions.

Modos:
    python check_drift.py --alerts-log api/alerts_log.jsonl
    python check_drift.py --simulate-drift      # Para demos/pruebas

Salidas (para GitHub Actions):
    - Escribe en $GITHUB_OUTPUT: drift_detected=true/false, score=<float>
    - Exit 0 siempre (el workflow decide si disparar retrain según el output)

Criterios de drift:
    CASO 2: Media del reconstruction_error en últimas N alertas sube >30%
            sobre la media histórica de entrenamiento.
    CASO 3: >5 alertas "critical" en la última hora.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Umbral de referencia del modelo v1.0.0 (se actualiza con model_metadata.json)
BASELINE_MSE_NORMAL = 0.03686  # reconstruction_mse_normal_mean del modelo actual
DRIFT_RELATIVE_THRESHOLD = 0.30   # Si la media sube >30% → drift
CRITICAL_ALERT_WINDOW_MINUTES = 60
CRITICAL_ALERT_COUNT_THRESHOLD = 5


def load_alerts(alerts_log_path: Path) -> list:
    if not alerts_log_path.exists():
        return []
    lines = alerts_log_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def check_drift_caso2(alerts: list) -> tuple[bool, float]:
    """
    CASO 2 — Drift en métricas:
    Calcula la media del reconstruction_error en las últimas 200 alertas
    y compara con la línea base del modelo entrenado.
    """
    if len(alerts) < 10:
        return False, 0.0

    recent = alerts[-200:]
    recent_errors = [a["reconstruction_error"] for a in recent]
    recent_mean = sum(recent_errors) / len(recent_errors)

    drift_score = (recent_mean - BASELINE_MSE_NORMAL) / BASELINE_MSE_NORMAL
    drift_detected = drift_score > DRIFT_RELATIVE_THRESHOLD

    print(f"[DRIFT-CASO2] Baseline MSE: {BASELINE_MSE_NORMAL:.5f}")
    print(f"[DRIFT-CASO2] Media reciente (últimas {len(recent)} alertas): {recent_mean:.5f}")
    print(f"[DRIFT-CASO2] Drift score: {drift_score:+.1%}  |  Umbral: +{DRIFT_RELATIVE_THRESHOLD:.0%}")
    print(f"[DRIFT-CASO2] {'🔴 DRIFT DETECTADO' if drift_detected else '🟢 Sin drift'}")

    return drift_detected, round(drift_score, 4)


def check_degradacion_caso3(alerts: list) -> tuple[bool, int]:
    """
    CASO 3 — Degradación de rendimiento:
    Cuenta alertas "critical" en la última hora.
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=CRITICAL_ALERT_WINDOW_MINUTES)

    recent_critical = []
    for a in alerts:
        try:
            ts = datetime.fromisoformat(a["timestamp"])
            if ts >= window_start and a.get("severity") == "critical":
                recent_critical.append(a)
        except (KeyError, ValueError):
            continue

    count = len(recent_critical)
    triggered = count >= CRITICAL_ALERT_COUNT_THRESHOLD

    print(f"[DRIFT-CASO3] Alertas críticas en la última hora: {count}  |  Umbral: {CRITICAL_ALERT_COUNT_THRESHOLD}")
    print(f"[DRIFT-CASO3] {'🔴 DEGRADACIÓN DETECTADA' if triggered else '🟢 Sin degradación'}")

    return triggered, count


def load_baseline_from_metadata(api_dir: Path):
    """Lee el MSE baseline directo del model_metadata.json actual."""
    global BASELINE_MSE_NORMAL
    meta_path = api_dir / "models" / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        baseline = meta.get("metrics", {}).get("reconstruction_mse_normal_mean")
        if baseline:
            BASELINE_MSE_NORMAL = float(baseline)
            print(f"[DRIFT] Baseline actualizado desde metadata: {BASELINE_MSE_NORMAL:.6f}")


def write_github_output(drift_detected: bool, score: float, reason: str):
    """Escribe variables de salida para GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"drift_detected={'true' if drift_detected else 'false'}\n")
            f.write(f"score={score}\n")
            f.write(f"reason={reason}\n")
    else:
        # En local, solo imprimir
        print(f"\n[OUTPUT] drift_detected={'true' if drift_detected else 'false'}")
        print(f"[OUTPUT] score={score}")
        print(f"[OUTPUT] reason={reason}")


def simulate_drift_data(alerts_log_path: Path):
    """
    Genera alertas sintéticas con DRIFT para demostración.
    Simula datos con temperatura +15% y velocidad -10%
    → reconstruction_error sube significativamente.
    """
    print("[SIMULATE] Generando alertas sintéticas con drift (temp+15%, rpm-10%)...")
    now = datetime.now(timezone.utc)
    synthetic_alerts = []

    import random
    random.seed(42)

    for i in range(50):
        # Datos normales históricos (las primeras 30)
        err = random.uniform(0.02, 0.06) if i < 30 else random.uniform(0.07, 0.15)
        severity = "critical" if err > 0.10 else "warning"
        ts = (now - timedelta(minutes=(50 - i) * 3)).isoformat()
        synthetic_alerts.append({
            "timestamp": ts,
            "machine_id": f"M-{(i % 5) + 1:03d}",
            "reconstruction_error": round(err, 5),
            "action": 2 if severity == "critical" else 1,
            "action_label": "Parada de emergencia" if severity == "critical" else "Mantenimiento preventivo",
            "severity": severity,
            "reading": {
                "Air temperature [K]": round(300 + (15 if i >= 30 else 0) + random.uniform(-1, 1), 2),
                "Process temperature [K]": round(310 + random.uniform(-1, 1), 2),
                "Rotational speed [rpm]": round(1500 * (0.9 if i >= 30 else 1.0) + random.uniform(-50, 50), 2),
                "Torque [Nm]": round(40 + random.uniform(-5, 5), 2),
                "Tool wear [min]": round(random.uniform(100, 200), 2),
                "_synthetic": True,
                "_drift_injected": i >= 30,
            },
        })

    alerts_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alerts_log_path, "w", encoding="utf-8") as f:
        for alert in synthetic_alerts:
            f.write(json.dumps(alert) + "\n")

    print(f"[SIMULATE] ✅ {len(synthetic_alerts)} alertas escritas en {alerts_log_path}")
    print(f"[SIMULATE]    Primeras 30: datos normales  |  Últimas 20: con drift +30%")
    return alerts_log_path


def main():
    parser = argparse.ArgumentParser(description="Detector de drift para MLOps pipeline")
    parser.add_argument("--alerts-log", type=str, default="api/alerts_log.jsonl")
    parser.add_argument("--simulate-drift", action="store_true",
                        help="Genera alertas sintéticas con drift para demo/pruebas")
    parser.add_argument("--api-dir", type=str,
                        default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    alerts_log_path = Path(args.alerts_log)
    api_dir = Path(args.api_dir)

    # Actualizar baseline desde el modelo en producción
    load_baseline_from_metadata(api_dir)

    if args.simulate_drift:
        alerts_log_path = simulate_drift_data(alerts_log_path)

    print(f"\n{'=' * 60}")
    print(f"MONITOR DE DRIFT — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'=' * 60}\n")

    alerts = load_alerts(alerts_log_path)
    print(f"[DRIFT] Alertas cargadas: {len(alerts)}")

    if not alerts:
        print("[DRIFT] Sin alertas registradas. Sistema operando normalmente.")
        write_github_output(False, 0.0, "no_alerts")
        return

    # Ejecutar los dos casos de detección
    drift_caso2, score_caso2 = check_drift_caso2(alerts)
    drift_caso3, count_caso3 = check_degradacion_caso3(alerts)

    drift_detected = drift_caso2 or drift_caso3
    score = max(score_caso2, count_caso3 / 10.0)

    reasons = []
    if drift_caso2:
        reasons.append(f"data_drift:{score_caso2:+.1%}")
    if drift_caso3:
        reasons.append(f"critical_alerts:{count_caso3}")
    reason_str = "+".join(reasons) if reasons else "none"

    print(f"\n{'=' * 60}")
    print(f"RESULTADO FINAL: {'🔴 DRIFT DETECTADO — se disparará reentrenamiento' if drift_detected else '🟢 SIN DRIFT — sistema estable'}")
    print(f"{'=' * 60}")

    write_github_output(drift_detected, round(score, 4), reason_str)


if __name__ == "__main__":
    main()
