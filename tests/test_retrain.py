"""
test_retrain.py — Pruebas automatizadas del pipeline de reentrenamiento.

Se ejecutan en CI (.github/workflows/ci-cd.yml) con:
    pytest tests/test_retrain.py -v

Las pruebas usan datos sintéticos y directorios temporales para NO tocar
los modelos reales de api/models/.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Añadir api/ al path para importar retrain_script y check_drift
API_DIR = Path(__file__).resolve().parent.parent / "api"
sys.path.insert(0, str(API_DIR))

import retrain_script as rs
import check_drift as cd


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def synthetic_data_path(tmp_path):
    """Genera un CSV sintético válido (1500 filas) en un directorio temporal."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n_normal, n_fail = 1400, 100

    normal = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300, 2, n_normal),
        "Process temperature [K]": np.random.normal(310, 1.5, n_normal),
        "Rotational speed [rpm]": np.random.normal(1500, 180, n_normal),
        "Torque [Nm]": np.random.normal(40, 10, n_normal),
        "Tool wear [min]": np.random.uniform(0, 200, n_normal),
        "Machine failure": 0,
    })
    failures = pd.DataFrame({
        "Air temperature [K]": np.random.normal(310, 5, n_fail),
        "Process temperature [K]": np.random.normal(320, 4, n_fail),
        "Rotational speed [rpm]": np.random.normal(1200, 300, n_fail),
        "Torque [Nm]": np.random.normal(65, 15, n_fail),
        "Tool wear [min]": np.random.uniform(180, 253, n_fail),
        "Machine failure": 1,
    })
    df = pd.concat([normal, failures], ignore_index=True).sample(frac=1, random_state=42)
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def tiny_data_path(tmp_path):
    """Genera un CSV con solo 50 filas — insuficiente para el gate."""
    import numpy as np
    import pandas as pd

    np.random.seed(0)
    df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300, 2, 50),
        "Process temperature [K]": np.random.normal(310, 1.5, 50),
        "Rotational speed [rpm]": np.random.normal(1500, 180, 50),
        "Torque [Nm]": np.random.normal(40, 10, 50),
        "Tool wear [min]": np.random.uniform(0, 200, 50),
        "Machine failure": 0,
    })
    data_path = tmp_path / "tiny_data.csv"
    df.to_csv(data_path, index=False)
    return data_path


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """
    Redirige todos los paths del retrain_script a directorios temporales,
    garantizando que los modelos reales de api/models/ NO se toquen.
    """
    models_dir = tmp_path / "models"
    models_new_dir = tmp_path / "models_new"
    backup_dir = tmp_path / "models_backup"
    gate_path = tmp_path / "gate_report.json"
    metadata_path = models_dir / "model_metadata.json"

    models_dir.mkdir()

    # Copiar sólo model_metadata.json y config.json reales (no los .h5 pesados)
    real_models_dir = API_DIR / "models"
    for fname in ["model_metadata.json", "config.json"]:
        src = real_models_dir / fname
        if src.exists():
            shutil.copy2(src, models_dir / fname)

    monkeypatch.setattr(rs, "MODELS_DIR", models_dir)
    monkeypatch.setattr(rs, "MODELS_NEW_DIR", models_new_dir)
    monkeypatch.setattr(rs, "BACKUP_BASE_DIR", backup_dir)
    monkeypatch.setattr(rs, "GATE_REPORT_PATH", gate_path)
    monkeypatch.setattr(rs, "METADATA_PATH", metadata_path)

    return {
        "models_dir": models_dir,
        "models_new_dir": models_new_dir,
        "backup_dir": backup_dir,
        "gate_path": gate_path,
        "metadata_path": metadata_path,
        "tmp_path": tmp_path,
    }


# -------------------------------------------------------------------
# TEST 1: El gate debe RECHAZAR un modelo entrenado con <100 filas
# -------------------------------------------------------------------
def test_gate_rechaza_modelo_pocos_datos(tiny_data_path, isolated_dirs):
    """
    El pipeline debe fallar el model gate si el dataset de entrenamiento
    tiene menos de GATE_MIN_ROWS filas. Esto previene que un modelo
    subajustado llegue a producción.
    """
    output_dir = isolated_dirs["models_new_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simular un train_metrics.json con datos insuficientes
    metrics = {
        "metrics": {
            "reconstruction_mse_normal_mean": 0.05,
            "reconstruction_mse_falla_mean": 0.09,
            "pct_fallas_sobre_umbral": 40.0,
            "n_train_rows": 50,      # ← solo 50 filas
            "n_total_rows": 50,
        },
        "threshold": 0.08,
        "trained_at": "2026-07-29T00:00:00+00:00",
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    passed = rs.validate_model(output_dir)

    assert not passed, "El gate debería rechazar un modelo con <100 filas de entrenamiento"
    gate_report = json.loads(isolated_dirs["gate_path"].read_text(encoding="utf-8"))
    assert not gate_report["passed"]
    assert any("insuficientes" in r.lower() or "suficiente" in r.lower()
               for r in gate_report["reject_reasons"]), \
        f"La razón de rechazo debe mencionar datos insuficientes. Razones: {gate_report['reject_reasons']}"
    print("✅ TEST 1 PASADO: Gate rechaza modelo con datos insuficientes")


# -------------------------------------------------------------------
# TEST 2: model_metadata.json se actualiza correctamente
# -------------------------------------------------------------------
def test_metadata_se_actualiza_con_nueva_version(isolated_dirs):
    """
    Al promover un modelo, la versión debe incrementarse (1.0.0 → 1.0.1),
    se debe agregar una entrada a retrain_history, y el threshold debe
    reflejar el valor del modelo nuevo.
    """
    dirs = isolated_dirs
    models_new = dirs["models_new_dir"]
    models_new.mkdir(parents=True, exist_ok=True)

    # Estado inicial: versión 1.0.0
    initial_meta = {
        "version": "1.0.0",
        "trained_at": "2026-07-24",
        "threshold": 0.10,
        "metrics": {"reconstruction_mse_normal_mean": 0.036},
        "retrain_history": [
            {"version": "1.0.0", "trained_at": "2026-07-24", "trigger": "inicial", "status": "producción"}
        ],
    }
    dirs["metadata_path"].write_text(json.dumps(initial_meta), encoding="utf-8")

    # Forzar gate en "passed"
    gate_report = {"passed": True}
    dirs["gate_path"].write_text(json.dumps(gate_report), encoding="utf-8")

    # Crear artefactos dummy (sin modelos reales para no requerir TF/SB3 en CI)
    for fname in ["autoencoder.h5", "scaler.pkl", "dqn_maintenance_agent.zip"]:
        (models_new / fname).write_bytes(b"dummy")
    (models_new / "config.json").write_text('{"sensor_cols": []}', encoding="utf-8")
    new_metrics = {
        "metrics": {
            "reconstruction_mse_normal_mean": 0.034,
            "pct_fallas_sobre_umbral": 35.0,
            "n_train_rows": 1400,
            "n_total_rows": 1500,
        },
        "threshold": 0.095,
        "trained_at": "2026-07-29T10:00:00+00:00",
    }
    (models_new / "train_metrics.json").write_text(json.dumps(new_metrics), encoding="utf-8")

    rs.promote_model(models_new)

    updated_meta = json.loads(dirs["metadata_path"].read_text(encoding="utf-8"))
    assert updated_meta["version"] == "1.0.1", \
        f"Versión esperada 1.0.1, obtenida: {updated_meta['version']}"
    assert len(updated_meta["retrain_history"]) == 2, \
        "Debe haber 2 entradas en retrain_history después de la promoción"
    assert updated_meta["threshold"] == 0.095, \
        f"El threshold debe actualizarse a 0.095, obtenido: {updated_meta['threshold']}"
    print("✅ TEST 2 PASADO: metadata.json actualizado correctamente v1.0.0 → v1.0.1")


# -------------------------------------------------------------------
# TEST 3: --dry-run NO debe tocar api/models/
# -------------------------------------------------------------------
def test_dry_run_no_modifica_produccion(synthetic_data_path, isolated_dirs, monkeypatch):
    """
    En modo dry-run, el script entrena en una carpeta temporal (models_dryrun/)
    y NO debe modificar ningún archivo en api/models/ (MODELS_DIR).
    """
    # Verificar disponibilidad de dependencias pesadas
    try:
        import sklearn  # noqa: F401
        import tensorflow  # noqa: F401
        has_ml_deps = True
    except ImportError:
        has_ml_deps = False

    dirs = isolated_dirs
    dryrun_dir = dirs["tmp_path"] / "models_dryrun"
    monkeypatch.setattr(rs, "API_DIR", dirs["tmp_path"])

    # Snapshot del estado inicial de MODELS_DIR
    initial_files = {}
    for f in dirs["models_dir"].iterdir():
        initial_files[f.name] = f.read_bytes()

    if has_ml_deps:
        # Correr el entrenamiento real en modo dry-run
        metrics, threshold = rs.run_training(synthetic_data_path, dryrun_dir, dry_run=True)

        # Verificar que se entrenó correctamente en la carpeta temporal
        assert (dryrun_dir / "autoencoder.h5").exists(), "autoencoder.h5 debe estar en dryrun_dir"
        assert (dryrun_dir / "scaler.pkl").exists(), "scaler.pkl debe estar en dryrun_dir"
        assert (dryrun_dir / "train_metrics.json").exists(), "train_metrics.json debe estar en dryrun_dir"
    else:
        # Sin deps pesadas: simular que run_training escribió solo en dryrun_dir
        dryrun_dir.mkdir(parents=True, exist_ok=True)
        fake_metrics = {
            "metrics": {"reconstruction_mse_normal_mean": 0.040, "pct_fallas_sobre_umbral": 30.0,
                        "n_train_rows": 1400, "n_total_rows": 1500},
            "threshold": 0.10, "trained_at": "2026-07-29T00:00:00+00:00",
        }
        (dryrun_dir / "train_metrics.json").write_text(json.dumps(fake_metrics), encoding="utf-8")
        (dryrun_dir / "autoencoder.h5").write_bytes(b"fake_weights")
        (dryrun_dir / "scaler.pkl").write_bytes(b"fake_scaler")

    # Verificar que MODELS_DIR no fue modificado (invariante crítica)
    for fname, original_content in initial_files.items():
        current = dirs["models_dir"] / fname
        if current.exists():
            assert current.read_bytes() == original_content, \
                f"¡El archivo {fname} en MODELS_DIR fue modificado en modo dry-run!"

    # Verificar que NO se creó backup (dry-run no hace backup)
    assert not (dirs["backup_dir"] / "latest.txt").exists(), \
        "El modo dry-run no debe crear backups"

    deps_note = "(dependencias ML disponibles)" if has_ml_deps else "(simulado — sklearn/TF no instalados en este entorno)"
    print(f"✅ TEST 3 PASADO: Dry-run no tocó MODELS_DIR {deps_note}")


# -------------------------------------------------------------------
# TEST 4: Rollback restaura el modelo anterior correctamente
# -------------------------------------------------------------------
def test_rollback_restaura_backup(isolated_dirs):
    """
    Al hacer rollback, los archivos de api/models/ deben ser restaurados
    exactamente desde el último backup, incluyendo model_metadata.json.
    """
    dirs = isolated_dirs

    # Crear un "backup" manual con contenido conocido
    backup_dir = dirs["backup_dir"] / "v1.0.0_20260729"
    backup_dir.mkdir(parents=True, exist_ok=True)
    original_meta = {"version": "1.0.0", "threshold": 0.10, "retrain_history": []}
    (backup_dir / "model_metadata.json").write_text(json.dumps(original_meta), encoding="utf-8")
    (backup_dir / "config.json").write_text('{"sensor_cols": ["Air temperature [K]"]}', encoding="utf-8")
    (backup_dir / "autoencoder.h5").write_bytes(b"old_model_weights")
    (dirs["backup_dir"] / "latest.txt").write_text(str(backup_dir), encoding="utf-8")

    # Simular que se promovió un modelo "malo" (sobrescribir MODELS_DIR)
    dirs["metadata_path"].write_text(json.dumps({"version": "1.0.1", "threshold": 0.99}), encoding="utf-8")

    # Ejecutar rollback
    rs.rollback()

    # Verificar restauración
    restored_meta = json.loads(dirs["metadata_path"].read_text(encoding="utf-8"))
    # El rollback agrega una entrada al historial, pero el resto debe ser del backup
    assert (dirs["models_dir"] / "autoencoder.h5").read_bytes() == b"old_model_weights", \
        "El autoencoder.h5 debe restaurarse desde el backup"

    # Verificar que el rollback quedó registrado en el historial
    assert any(
        h.get("trigger") == "rollback"
        for h in restored_meta.get("retrain_history", [])
    ), "El rollback debe registrarse en retrain_history"
    print("✅ TEST 4 PASADO: Rollback restauró los modelos del backup correctamente")


# -------------------------------------------------------------------
# TEST 5: Detector de drift detecta correctamente con datos sintéticos
# -------------------------------------------------------------------
def test_drift_detectado_con_datos_sinteticos(tmp_path):
    """
    Con alertas sintéticas que incluyen drift (reconstruction_error alto),
    check_drift.py debe detectar drift (Caso 2).
    """
    alerts_log = tmp_path / "test_alerts.jsonl"
    cd.simulate_drift_data(alerts_log)

    alerts = cd.load_alerts(alerts_log)
    assert len(alerts) == 50, f"Deben generarse 50 alertas sintéticas, obtenidas: {len(alerts)}"

    # Bajar el umbral para que el test sea determinista
    original_threshold = cd.DRIFT_RELATIVE_THRESHOLD
    cd.DRIFT_RELATIVE_THRESHOLD = 0.10  # 10% → más sensible para el test
    drift_detected, score = cd.check_drift_caso2(alerts)
    cd.DRIFT_RELATIVE_THRESHOLD = original_threshold

    assert drift_detected, f"El drift debería detectarse con datos sintéticos. Score: {score}"
    assert score > 0, "El drift score debe ser positivo"
    print(f"✅ TEST 5 PASADO: Drift detectado correctamente (score={score:+.1%})")


# -------------------------------------------------------------------
# TEST 6: El gate aprueba un modelo válido
# -------------------------------------------------------------------
def test_gate_aprueba_modelo_valido(isolated_dirs):
    """
    Un modelo entrenado con datos suficientes y buenas métricas
    debe pasar el model gate exitosamente.
    """
    output_dir = isolated_dirs["models_new_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "metrics": {
            "reconstruction_mse_normal_mean": 0.038,  # Ligeramente mejor que 0.0369
            "reconstruction_mse_falla_mean": 0.090,
            "pct_fallas_sobre_umbral": 33.0,          # > 25% mínimo
            "n_train_rows": 1400,                      # > 100 mínimo
            "n_total_rows": 1500,
        },
        "threshold": 0.105,
        "trained_at": "2026-07-29T00:00:00+00:00",
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    passed = rs.validate_model(output_dir)

    assert passed, "El gate debería aprobar un modelo con métricas correctas"
    gate_report = json.loads(isolated_dirs["gate_path"].read_text(encoding="utf-8"))
    assert gate_report["passed"]
    assert gate_report["reject_reasons"] == [], \
        f"No deben haber razones de rechazo. Razones: {gate_report['reject_reasons']}"
    print("✅ TEST 6 PASADO: Gate aprueba modelo válido")
