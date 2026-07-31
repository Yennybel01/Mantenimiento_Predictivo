import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui, i18n

ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow(i18n.t("eyebrow_monitoreo"))
st.markdown(f"# {i18n.t('title_monitoreo')}")
st.caption(i18n.t("subtitle_monitoreo"))

if health is None:
    st.stop()

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
RANGES = {
    "Air temperature [K]": (295.0, 304.0),
    "Process temperature [K]": (305.0, 313.5),
    "Rotational speed [rpm]": (1180, 2100),
    "Torque [Nm]": (25.0, 55.0),
    "Tool wear [min]": (0, 253),
}

import numpy as np


def _simulate_reading(anomaly: bool, machine_id: str) -> dict:
    row = {"machine_id": machine_id}
    for feat, (lo, hi) in RANGES.items():
        base = np.random.uniform(lo, hi)
        if anomaly:
            shift = (hi - lo) * np.random.uniform(0.4, 0.9) * np.random.choice([-1, 1])
            base += shift
        row[feat] = float(base)
    row["tiempo_desde_mantenimiento_min"] = float(np.random.uniform(0, 220))
    return row


if "monitor_history" not in st.session_state:
    st.session_state.monitor_history = pd.DataFrame()

# --------------------------------------------------------------------------
# Controles
# --------------------------------------------------------------------------
ui.panel_start(i18n.t("panel_data_source"), i18n.t("panel_data_source_sub"))
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
with c1:
    machine_id = st.text_input(i18n.t("machine_id"), "M-001")
with c2:
    n_points = st.number_input(i18n.t("readings_to_generate"), 1, 50, 5)
with c3:
    anomaly_prob = st.slider(i18n.t("anomaly_prob"), 0.0, 0.6, 0.15)
with c4:
    gen = st.button(i18n.t("btn_generate"), use_container_width=True)

upload = st.file_uploader(i18n.t("csv_upload_text").format(cols=', '.join(FEATURES)), type=["csv"])
ui.panel_end()

if gen:
    new_rows = [_simulate_reading(np.random.rand() < anomaly_prob, machine_id) for _ in range(n_points)]
    with st.spinner(i18n.t("querying_api")):
        results = api_client.predict(new_rows, api_url)
    for row, res in zip(new_rows, results):
        row.update(res)
        row["timestamp"] = datetime.now()
    st.session_state.monitor_history = pd.concat(
        [st.session_state.monitor_history, pd.DataFrame(new_rows)], ignore_index=True
    ).tail(300)
    api_client.get_alerts.clear()

if upload is not None:
    uploaded_df = pd.read_csv(upload)
    missing = [f for f in FEATURES if f not in uploaded_df.columns]
    if missing:
        st.error(f"{i18n.t('missing_cols')}: {missing}")
    else:
        if "machine_id" not in uploaded_df.columns:
            uploaded_df["machine_id"] = machine_id
        if "tiempo_desde_mantenimiento_min" not in uploaded_df.columns:
            uploaded_df["tiempo_desde_mantenimiento_min"] = 0.0
        records = uploaded_df.to_dict(orient="records")
        with st.spinner(i18n.t("querying_api")):
            results = api_client.predict(records, api_url)
        for row, res in zip(records, results):
            row.update(res)
            row["timestamp"] = datetime.now()
        st.session_state.monitor_history = pd.concat(
            [st.session_state.monitor_history, pd.DataFrame(records)], ignore_index=True
        ).tail(300)
        api_client.get_alerts.clear()

history = st.session_state.monitor_history

if history.empty:
    st.info(i18n.t("monitoreo_start_info"))
    st.stop()

# --------------------------------------------------------------------------
# KPIs de la sesión
# --------------------------------------------------------------------------
n_total = len(history)
n_anom = int(history["is_anomaly"].sum())
n_crit = int((history["severity"] == "critical").sum())
cards = [
    ui.kpi_card(i18n.t("kpi_screen_readings"), str(n_total), i18n.t("kpi_screen_readings_sub"), "var(--accent-signal)"),
    ui.kpi_card(i18n.t("kpi_above_threshold"), str(n_anom), i18n.t("kpi_above_threshold_sub").format(pct=n_anom/n_total*100), "var(--status-warning)"),
    ui.kpi_card(i18n.t("kpi_emergency_stops"), str(n_crit), i18n.t("kpi_emergency_stops_sub"), "var(--status-critical)"),
]
ui.kpi_grid(cards)

# --------------------------------------------------------------------------
# Gráfico tipo osciloscopio
# --------------------------------------------------------------------------
ui.panel_start(i18n.t("panel_reconstruction_error"), i18n.t("panel_reconstruction_error_sub"))
metadata = api_client.get_metadata(api_url) or {}
threshold = metadata.get("threshold", 0.35)

color_map = {"ok": "#0D9488", "warning": "#D97706", "critical": "#E11D48"}
colors = history["severity"].map(color_map).fillna("#0D9488")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(history))), y=history["reconstruction_error"],
    mode="lines", line=dict(color="#2563EB", width=1.5), name=i18n.t("chart_error"), showlegend=False,
))
fig.add_trace(go.Scatter(
    x=list(range(len(history))), y=history["reconstruction_error"],
    mode="markers", marker=dict(color=colors, size=7, line=dict(width=1, color="#FFFFFF")),
    name=i18n.t("chart_reading"), showlegend=False,
))
fig.add_hline(y=threshold, line_dash="dot", line_color="#64748B", annotation_text=i18n.t("chart_threshold"), annotation_font_color="#64748B")
fig.update_layout(
    height=340, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(color="#64748B", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#E2E8F0", title=i18n.t("chart_reading")),
    yaxis=dict(gridcolor="#E2E8F0", title=i18n.t("chart_error")),
)
st.plotly_chart(fig, use_container_width=True)
ui.panel_end()

# --------------------------------------------------------------------------
# Tabla
# --------------------------------------------------------------------------
ui.panel_start(i18n.t("panel_log_readings"), "")
display_cols = ["timestamp", "machine_id"] + FEATURES + ["reconstruction_error", "action_label", "severity"]
# Mapear las etiquetas de acción a su idioma correspondiente
display_df = history.copy()
if "action_label" in display_df.columns:
    display_df["action_label"] = display_df["action_label"].apply(lambda x: i18n.t(x, x))

display_cols = [c for c in display_cols if c in display_df.columns]
st.dataframe(
    display_df[display_cols].sort_values("timestamp", ascending=False),
    use_container_width=True, height=320,
)
if st.button(i18n.t("btn_clear_history")):
    st.session_state.monitor_history = pd.DataFrame()
    st.rerun()
ui.panel_end()
