import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import api_client, ui

st.set_page_config(
    page_title="Predictive Maintenance — Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.load_css()

# --------------------------------------------------------------------------
# Sidebar: configuración de conexión a la API
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="pm-eyebrow">Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Control Room")
    st.caption("Autoencoder + Agente DQN")
    st.markdown("---")
    api_url = st.text_input("API URL", value=api_client.get_api_url())
    st.session_state["api_url"] = api_url
    st.markdown("---")
    st.caption(
        "Este dashboard consume la API REST del backend. "
        "Si el equipo de deploy cambia el host de producción, "
        "actualiza la URL aquí."
    )

health = api_client.check_health(api_url)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
ui.eyebrow("SISTEMA · MANTENIMIENTO PREDICTIVO")
st.markdown("# Vista general")
st.caption(
    "Autoencoder para detección de anomalías + Agente de Aprendizaje por Refuerzo "
    "(DQN) para recomendar acciones de mantenimiento, en producción."
)

is_online = ui.api_status_banner(health, api_url)

if not is_online:
    st.info(
        "Para levantar el backend: `cd api && uvicorn main:app --reload --port 8000`\n\n"
        "Luego recarga esta página."
    )
    st.stop()

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
metadata = api_client.get_metadata(api_url) or {}
alerts = api_client.get_alerts(api_url, limit=200) or []
n_warning = len([a for a in alerts if a["severity"] == "warning"])
n_critical = len([a for a in alerts if a["severity"] == "critical"])

cards = [
    ui.kpi_card("Versión del modelo", metadata.get("version", "—"), metadata.get("status", ""), "var(--accent-signal)"),
    ui.kpi_card("Umbral de anomalía", f"{metadata.get('threshold', 0):.4f}", "error de reconstrucción", "var(--accent-signal)"),
    ui.kpi_card("Alertas — atención", str(n_warning), "mantenimiento preventivo", "var(--status-warning)"),
    ui.kpi_card("Alertas — críticas", str(n_critical), "parada de emergencia", "var(--status-critical)"),
]
ui.kpi_grid(cards)

# --------------------------------------------------------------------------
# Navegación rápida
# --------------------------------------------------------------------------
ui.panel_start("Módulos del sistema", "Navega usando el menú lateral")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/1_📡_Monitoreo.py", label="📡  Monitoreo", help="Lecturas de sensores en tiempo real")
with c2:
    st.page_link("pages/2_📊_Análisis.py", label="📊  Análisis", help="Métricas del modelo y MLOps")
with c3:
    st.page_link("pages/3_🔧_Predicciones.py", label="🔧  Predicciones", help="Evaluar una lectura puntual")
with c4:
    st.page_link("pages/4_🚨_Alertas.py", label="🚨  Alertas", help="Historial de eventos que requieren acción")
ui.panel_end()

# --------------------------------------------------------------------------
# Resumen de arquitectura (útil para el equipo que da mantenimiento)
# --------------------------------------------------------------------------
ui.panel_start("Arquitectura", "Separación backend / frontend para facilitar el mantenimiento")
st.markdown(
    """```
api/        FastAPI — única capa que toca los modelos (.h5, .pkl, DQN .zip)
app/        Streamlit — solo consume la API vía HTTP, nunca carga modelos
.github/    Workflow de CI/CD para reentrenamiento periódico
```
**Contrato de la API** (`/docs` para la referencia interactiva de FastAPI):
`GET /health` · `GET /model/metadata` · `POST /predict` · `GET /alerts` · `POST /retrain`"""
)
ui.panel_end()
