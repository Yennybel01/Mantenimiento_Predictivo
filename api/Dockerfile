FROM python:3.11-slim

WORKDIR /app

# build-essential: requerido por algunas ruedas de numpy/scikit-learn/tensorflow-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar en pasos separados (no todo junto): en builds con poca memoria
# (ej. Render free tier, 512MB), pip resolviendo tensorflow-cpu +
# stable-baselines3 + gymnasium + scikit-learn a la vez puede superar el
# límite y matar el build con OOM. Instalando de a poco, cada paso libera
# la memoria del anterior antes de seguir.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy joblib scikit-learn
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic requests
RUN pip install --no-cache-dir tensorflow-cpu
RUN pip install --no-cache-dir gymnasium

# IMPORTANTE: instalar PyTorch CPU-only ANTES de stable-baselines3. Si no,
# pip instala por defecto la variante con CUDA completa (varios GB extra de
# librerías NVIDIA que nunca se usan en Render, sin GPU), y eso satura la
# memoria RAM al importarla en tiempo de ejecución -> OOM en el arranque.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir stable-baselines3

COPY . .

EXPOSE 8000

# Render inyecta la variable $PORT; localmente (docker-compose) cae en 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]