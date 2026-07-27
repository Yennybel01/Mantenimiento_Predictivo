import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui

st.set_page_config(page_title="Predicciones · Predictive Maintenance", page_icon="🔧", layout="wide")
ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow("MÓDULO · INFERENCIA PUNTUAL")
st.markdown("# 🔧 Evaluar una lectura")
st.caption("Ingresa los valores de sensores de una máquina y consulta al Autoencoder + Agente DQN qué acción recomienda.")

if health is None:
    st.stop()

ui.panel_start("Lectura de sensores", "")
c1, c2 = st.columns(2)
with c1:
    air_temp = st.number_input("Air temperature [K]", 290.0, 320.0, 300.0)
    process_temp = st.number_input("Process temperature [K]", 295.0, 320.0, 310.0)
    rot_speed = st.number_input("Rotational speed [rpm]", 800, 3000, 1500)
    tiempo_mantenimiento = st.number_input(
        "Tiempo desde último mantenimiento (min)", 0, 500, 0,
        help="Usado por el agente DQN para decidir la acción.",
    )
with c2:
    torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0)
    tool_wear = st.number_input("Tool wear [min]", 0, 260, 50)
    machine_id = st.text_input("Machine ID", value="M-001")

submitted = st.button("🔍 Predecir", use_container_width=True, type="primary")
ui.panel_end()

if submitted:
    record = {
        "machine_id": machine_id,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "tiempo_desde_mantenimiento_min": tiempo_mantenimiento,
    }
    with st.spinner("Consultando la API…"):
        try:
            result = api_client.predict([record], api_url)[0]
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.stop()
    api_client.get_alerts.clear()

    metadata = api_client.get_metadata(api_url) or {}
    threshold = metadata.get("threshold", 0.35)

    colL, colR = st.columns([1, 1])
    with colL:
        ui.panel_start("Resultado", f"Machine: {machine_id}")
        st.markdown(ui.status_pill(result["severity"], result["action_label"].upper()), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Error de reconstrucción", f"{result['reconstruction_error']:.4f}")
        st.metric("Umbral", f"{threshold:.4f}")
        st.write(f"**¿Supera el umbral?** {'Sí' if result['is_anomaly'] else 'No'}")
        with st.expander("¿Qué significa cada acción?"):
            st.markdown(
                "- **Operar con normalidad**: el agente no ve riesgo suficiente.\n"
                "- **Mantenimiento preventivo**: hay señales de desgaste/anomalía, conviene revisar pronto.\n"
                "- **Parada de emergencia**: alto riesgo de falla inminente, detener la máquina."
            )
        ui.panel_end()

    with colR:
        ui.panel_start("Error vs. umbral", "")
        color = {"ok": "#0D9488", "warning": "#D97706", "critical": "#E11D48"}[result["severity"]]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["reconstruction_error"],
            number={"font": {"color": "#0F172A", "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, max(result["reconstruction_error"] * 1.4, threshold * 2)], "tickcolor": "#64748B"},
                "bar": {"color": color},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "threshold": {"line": {"color": "#0F172A", "width": 3}, "thickness": 0.8, "value": threshold},
            },
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="#FFFFFF", font=dict(color="#64748B"))
        st.plotly_chart(fig, use_container_width=True)
        ui.panel_end()
