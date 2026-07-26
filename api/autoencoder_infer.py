"""
Script de inferencia del Autoencoder, pensado para correr en un SUBPROCESO
separado del proceso principal de Streamlit.

Por qué: TensorFlow (usado por el Autoencoder) y PyTorch (usado por
stable-baselines3 para el agente DQN) chocan al importarse juntos en el
mismo proceso -> segmentation fault. Ejecutando el Autoencoder aquí, en un
proceso aparte, evitamos el conflicto sin sacrificar ninguno de los dos
modelos.

Uso:
  echo '{"sensor_cols": [...], "rows": [[...], [...]]}' | python3 autoencoder_infer.py
  -> imprime en stdout: {"errors": [0.01, 0.23, ...]}
"""

import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    payload = json.loads(sys.stdin.read())
    sensor_cols = payload["sensor_cols"]
    rows = payload["rows"]

    import numpy as np
    import joblib
    from tensorflow import keras

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    autoencoder = keras.models.load_model(MODELS_DIR / "autoencoder.h5", compile=False)

    X = scaler.transform(np.array(rows))
    X_rec = autoencoder.predict(X, verbose=0)
    errors = np.mean((X - X_rec) ** 2, axis=1)

    print(json.dumps({"errors": errors.tolist()}))


if __name__ == "__main__":
    main()
