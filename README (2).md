# Predictive Maintenance — UI (Streamlit)

Interfaz para el sistema de mantenimiento predictivo basado en Autoencoder
(detección de anomalías no supervisada). Este módulo corresponde a la parte
de **IU** del proyecto, diseñado para integrarse con los módulos de
**train**, **deploy** y **MLOps** sin necesidad de reescribir código.

## Cómo correrla

```bash
cd pm_ui
pip install -r requirements.txt
streamlit run app.py
```

Ya incluye los artefactos reales entregados por el equipo de train
(`models/autoencoder.h5`, `models/scaler.pkl`, `models/config.json`,
`models/dqn_maintenance_agent.zip`), así que corre con el sistema real:
**Autoencoder** (detecta anomalías por error de reconstrucción) +
**Agente DQN** (recomienda una acción: operar / mantenimiento preventivo /
parada de emergencia). Si por algún motivo faltan esos archivos, la UI cae
automáticamente a un modo demo con PCA y datos simulados.

### ⚠️ Nota técnica importante: TensorFlow + PyTorch en el mismo proceso

TensorFlow (usado por el Autoencoder) y PyTorch (usado por
`stable-baselines3` para el agente DQN) **chocan si se importan juntos en
el mismo proceso de Python y provocan un `segmentation fault`**. Por eso,
el Autoencoder se ejecuta en un **subproceso aislado**
(`utils/autoencoder_infer.py`), mientras que el agente DQN se mantiene en
el proceso principal de Streamlit. Esto añade ~1-2 segundos de overhead por
predicción (arranque de TensorFlow en el subproceso), pero evita el crash.
No hace falta tocar nada de esto al usar la app — ya está resuelto — pero
es bueno saberlo si en el futuro se refactoriza `model_utils.py`.

## Vistas

1. **📊 Monitoreo en tiempo real** — simula/recibe lecturas de sensores,
   grafica el error de reconstrucción y marca anomalías.
2. **🔧 Predicción individual** — formulario para una lectura puntual de un
   motor, muestra el error de reconstrucción vs. el umbral.
3. **📈 Dashboard MLOps** — versión del modelo, hiperparámetros, métricas,
   historial de reentrenamientos y botón para disparar un reentrenamiento.

## Cómo se conecta con el resto del equipo

### 1) Con el equipo de **train**
Cuando tengan el modelo entrenado, colocar en `models/`:
- `autoencoder.h5` (o `.keras`)
- `scaler.pkl` (el `StandardScaler`/`MinMaxScaler` usado antes de entrenar)
- Actualizar `models/model_metadata.json` con: `threshold`, `metrics`,
  `hyperparameters`, `trained_at`.

La UI detecta automáticamente estos archivos y deja de usar el modelo demo.
**No hay que tocar `app.py`.**

Features esperadas, en este orden:
```
Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min]
```

### 2) Con el equipo de **deploy**
Cuando la API esté lista, solo hay que pegar la URL en la barra lateral
(campo "URL de la API de predicción"). La UI espera este contrato REST:

```
GET  /health
     -> 200 OK  (para saber si la API está viva)

POST /predict
     body: {"records": [{ "Air temperature [K]": .., "Process temperature [K]": ..,
                            "Rotational speed [rpm]": .., "Torque [Nm]": .., "Tool wear [min]": .. }, ...]}
     -> {"results": [{"reconstruction_error": 0.42, "is_anomaly": true}, ...]}

GET  /model/metadata
     -> mismo esquema que models/model_metadata.json

POST /retrain
     -> dispara el pipeline de reentrenamiento (CI/CD) y responde con un
        identificador de ejecución, ej: {"run_id": "...", "status": "started"}
```

### 3) Con el equipo de **MLOps**
El dashboard MLOps lee `models/model_metadata.json` (o `/model/metadata`
en la API). Cada vez que el pipeline de reentrenamiento corra, debe:
- Registrar una nueva entrada en `retrain_history`.
- Actualizar `version`, `trained_at`, `metrics` y `threshold`.

Así el dashboard siempre refleja el estado real del modelo en producción,
sin cambios en la UI.

## Notas de diseño

- La función `predict()` en `utils/model_utils.py` implementa un *fallback*
  de 3 niveles: **API real → modelo local real → modelo demo**. Esto
  garantiza que la app nunca se cae, incluso si un módulo del equipo aún no
  está listo.
- Todo el texto orientado a la explicación técnica del funcionamiento de la
  app está en inglés (requisito de la exposición); las etiquetas de UI para
  el usuario final están en español.
