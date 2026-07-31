import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui, i18n

ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow(i18n.t("eyebrow_alertas"))
st.markdown(f"# {i18n.t('title_alertas')}")
st.caption(i18n.t("subtitle_alertas"))

if health is None:
    st.stop()

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    severity_filter = st.selectbox(i18n.t("filter_severity"), [i18n.t("all_filter"), "warning", "critical"])
with c2:
    limit = st.number_input(i18n.t("max_to_show"), 10, 500, 100)
with c3:
    if st.button(i18n.t("btn_update")):
        api_client.get_alerts.clear()

sev = None if severity_filter == i18n.t("all_filter") else severity_filter
alerts = api_client.get_alerts(api_url, limit=int(limit), severity=sev) or []

n_warning = len([a for a in alerts if a["severity"] == "warning"])
n_critical = len([a for a in alerts if a["severity"] == "critical"])
cards = [
    ui.kpi_card(i18n.t("kpi_total_view"), str(len(alerts)), "", "var(--accent-signal)"),
    ui.kpi_card(i18n.t("kpi_preventive_maintenance"), str(n_warning), "", "var(--status-warning)"),
    ui.kpi_card(i18n.t("kpi_emergency_stops_alertas"), str(n_critical), "", "var(--status-critical)"),
]
ui.kpi_grid(cards)

ui.panel_start(i18n.t("panel_events"), i18n.t("panel_events_sub"))
if not alerts:
    st.info(i18n.t("no_alerts"))
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
    with st.expander(i18n.t("expander_technical_details")):
        import pandas as pd
        rows = []
        for a in alerts:
            row = {"timestamp": a["timestamp"], "machine_id": a["machine_id"],
                   "severity": a["severity"], "error": a["reconstruction_error"]}
            row.update(a.get("reading", {}))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
