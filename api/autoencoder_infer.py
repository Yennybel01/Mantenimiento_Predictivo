"""
Script de inferencia del Autoencoder, pensado para correr como un
SUBPROCESO PERSISTENTE (un único proceso vivo durante toda la vida de la
API), separado del proceso principal.

Por qué separado: TensorFlow (usado por el Autoencoder) y PyTorch (usado
por stable-baselines3 para el agente DQN) chocan al importarse juntos en
el mismo proceso -> segmentation fault.

Por qué PERSISTENTE (y no un subproceso nuevo por request, como en la
versión anterior): reimportar TensorFlow completo en cada predicción tarda
varios segundos en local y puede superar los 60s en entornos con poca CPU
(ej. Render free tier), causando timeouts. Cargando el modelo UNA sola vez
y dejando el proceso vivo, cada predicción posterior solo paga el costo de
una inferencia (milisegundos), no de un arranque de TensorFlow completo.

Protocolo: el proceso padre (ml_core.py) escribe una línea JSON por stdin
y lee una línea JSON de respuesta por stdout, por cada predicción. Nunca
cierra el proceso entre requests.

Entrada por línea:  {"sensor_cols": [...], "rows": [[...], [...]]}
Salida por línea:   {"errors": [0.01, 0.23, ...]}
"""

import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    import numpy as np
    import joblib
    from tensorflow import keras

    # Carga unica del modelo y el scaler al arrancar el proceso.
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    autoencoder = keras.models.load_model(MODELS_DIR / "autoencoder.h5", compile=False)

    # Aviso al padre de que ya esta listo para recibir predicciones.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            rows = payload["rows"]
            X = scaler.transform(np.array(rows))
            X_rec = autoencoder.predict(X, verbose=0)
            errors = np.mean((X - X_rec) ** 2, axis=1)
            print(json.dumps({"errors": errors.tolist()}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()