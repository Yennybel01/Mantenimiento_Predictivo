import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui, i18n

ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow(i18n.t("eyebrow_predicciones"))
st.markdown(f"# {i18n.t('title_predicciones')}")
st.caption(i18n.t("subtitle_predicciones"))

if health is None:
    st.stop()

ui.panel_start(i18n.t("panel_sensor_readings"), "")
c1, c2 = st.columns(2)
with c1:
    air_temp = st.number_input("Air temperature [K]", 290.0, 320.0, 300.0)
    process_temp = st.number_input("Process temperature [K]", 295.0, 320.0, 310.0)
    rot_speed = st.number_input("Rotational speed [rpm]", 800, 3000, 1500)
    tiempo_mantenimiento = st.number_input(
        i18n.t("time_since_maintenance"), 0, 500, 0,
        help=i18n.t("time_since_maintenance_help"),
    )
with c2:
    torque = st.number_input("Torque [Nm]", 0.0, 80.0, 40.0)
    tool_wear = st.number_input("Tool wear [min]", 0, 260, 50)
    machine_id = st.text_input(i18n.t("machine_id"), value="M-001")

submitted = st.button(i18n.t("btn_predict"), use_container_width=True, type="primary")
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
    with st.spinner(i18n.t("querying_api")):
        try:
            result = api_client.predict([record], api_url)[0]
        except Exception as e:
            st.error(i18n.t("predict_error").format(error=e))
            st.stop()
    api_client.get_alerts.clear()

    metadata = api_client.get_metadata(api_url) or {}
    threshold = metadata.get("threshold", 0.35)

    colL, colR = st.columns([1, 1])
    with colL:
        ui.panel_start(i18n.t("panel_result"), f"{i18n.t('machine_label')}: {machine_id}")
        translated_label = i18n.t(result["action_label"], result["action_label"]).upper()
        st.markdown(ui.status_pill(result["severity"], translated_label), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(i18n.t("chart_error"), f"{result['reconstruction_error']:.4f}")
        st.metric(i18n.t("kpi_threshold_analisis"), f"{threshold:.4f}")
        
        is_above = i18n.t("yes") if result['is_anomaly'] else i18n.t("no")
        st.write(f"**{i18n.t('is_above_threshold')}** {is_above}")
        
        with st.expander(i18n.t("action_explanation_title")):
            st.markdown(i18n.t("action_explanation_content"))
        ui.panel_end()

    with colR:
        ui.panel_start(i18n.t("panel_error_vs_threshold"), "")
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
