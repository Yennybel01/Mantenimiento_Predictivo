import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui

st.set_page_config(page_title="Alertas · Predictive Maintenance", page_icon="🚨", layout="wide")
ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow("MÓDULO · ALERTAS")
st.markdown("# 🚨 Historial de alertas")
st.caption("Cada vez que el agente recomienda mantenimiento preventivo o parada de emergencia, queda registrado aquí para que el equipo de operaciones le dé seguimiento.")

if health is None:
    st.stop()

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    severity_filter = st.selectbox("Filtrar por severidad", ["Todas", "warning", "critical"])
with c2:
    limit = st.number_input("Máximo a mostrar", 10, 500, 100)
with c3:
    if st.button("🔄 Actualizar"):
        api_client.get_alerts.clear()

sev = None if severity_filter == "Todas" else severity_filter
alerts = api_client.get_alerts(api_url, limit=int(limit), severity=sev) or []

n_warning = len([a for a in alerts if a["severity"] == "warning"])
n_critical = len([a for a in alerts if a["severity"] == "critical"])
cards = [
    ui.kpi_card("Total en vista", str(len(alerts)), "", "var(--accent-signal)"),
    ui.kpi_card("Mantenimiento preventivo", str(n_warning), "", "var(--status-warning)"),
    ui.kpi_card("Paradas de emergencia", str(n_critical), "", "var(--status-critical)"),
]
ui.kpi_grid(cards)

ui.panel_start("Eventos", "Ordenados del más reciente al más antiguo")
if not alerts:
    st.info("No hay alertas registradas todavía. Genera lecturas en el módulo de Monitoreo o Predicciones.")
else:
    row_color = {"warning": "var(--status-warning)", "critical": "var(--status-critical)"}
    for a in alerts:
        ts = a["timestamp"].replace("T", " ").split(".")[0]
        st.markdown(
            ui.alert_row(ts, a["machine_id"], a["severity"], a["action_label"], a["reconstruction_error"]),
            unsafe_allow_html=True,
        )
ui.panel_end()

if alerts:
    with st.expander("Ver detalle de lecturas (para diagnóstico técnico)"):
        import pandas as pd
        rows = []
        for a in alerts:
            row = {"timestamp": a["timestamp"], "machine_id": a["machine_id"],
                   "severity": a["severity"], "error": a["reconstruction_error"]}
            row.update(a.get("reading", {}))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
