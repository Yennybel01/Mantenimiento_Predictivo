import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import api_client, ui, i18n

ui.load_css()

api_url = api_client.get_api_url()
health = api_client.check_health(api_url)

ui.eyebrow(i18n.t("eyebrow_analisis"))
st.markdown(f"# {i18n.t('title_analisis')}")
st.caption(i18n.t("subtitle_analisis"))

if health is None:
    st.stop()

metadata = api_client.get_metadata(api_url) or {}
metrics = metadata.get("metrics", {})
hparams = metadata.get("hyperparameters", {})

# --------------------------------------------------------------------------
# KPIs principales
# --------------------------------------------------------------------------
cards = [
    ui.kpi_card(i18n.t("kpi_version_analisis"), metadata.get("version", "—"), metadata.get("framework", ""), "var(--accent-signal)"),
    ui.kpi_card(i18n.t("kpi_dataset"), "AI4I 2020", metadata.get("dataset", "")[:28] + "…", "var(--accent-signal)"),
    ui.kpi_card(i18n.t("kpi_trained"), metadata.get("trained_at", "—"), i18n.t("kpi_trained_sub"), "var(--status-ok)"),
    ui.kpi_card(i18n.t("kpi_threshold_analisis"), f"{metadata.get('threshold', 0):.4f}", i18n.t("kpi_threshold_analisis_sub"), "var(--status-warning)"),
]
ui.kpi_grid(cards)

# --------------------------------------------------------------------------
# Distribución del error de reconstrucción (normal vs falla)
# --------------------------------------------------------------------------
col1, col2 = st.columns([1.3, 1])
with col1:
    ui.panel_start(i18n.t("panel_ae_separation"), i18n.t("panel_ae_separation_sub"))
    normal_mean = metrics.get("reconstruction_mse_normal_mean")
    falla_mean = metrics.get("reconstruction_mse_falla_mean")
    if normal_mean is not None and falla_mean is not None:
        fig = go.Figure(go.Bar(
            x=[i18n.t("bar_normal"), i18n.t("bar_falla")], y=[normal_mean, falla_mean],
            marker_color=["#0D9488", "#E11D48"], width=0.5,
            text=[f"{normal_mean:.4f}", f"{falla_mean:.4f}"], textposition="outside",
            textfont=dict(color="#0F172A", family="IBM Plex Mono"),
        ))
        threshold = metadata.get("threshold")
        if threshold:
            fig.add_hline(y=threshold, line_dash="dot", line_color="#64748B",
                           annotation_text=f"{i18n.t('chart_threshold')} ({threshold:.4f})", annotation_font_color="#64748B")
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#64748B", family="IBM Plex Mono"),
            yaxis=dict(gridcolor="#E2E8F0", title="MSE"),
            xaxis=dict(gridcolor="#E2E8F0"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        pct = metrics.get("pct_fallas_sobre_umbral")
        if pct is not None:
            st.caption(i18n.t("recall_caption").format(pct=pct))
    else:
        st.info(i18n.t("no_metrics"))
    ui.panel_end()

with col2:
    ui.panel_start(i18n.t("panel_dqn_agent"), i18n.t("panel_dqn_agent_sub"))
    reaccion = metrics.get("dqn_episodios_reaccion_a_tiempo")
    if reaccion:
        try:
            done, total = [int(x.strip()) for x in reaccion.split("/")]
            frac = done / total
        except Exception:
            frac = 0.5
            done, total = "?", "?"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=frac * 100,
            number={"suffix": "%", "font": {"color": "#0F172A", "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": "#2563EB"},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(225, 29, 72, 0.15)"},
                    {"range": [40, 70], "color": "rgba(217, 119, 6, 0.15)"},
                    {"range": [70, 100], "color": "rgba(13, 148, 136, 0.15)"},
                ],
            },
        ))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="#FFFFFF", font=dict(color="#64748B"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(i18n.t("dqn_reaction_caption").format(done=done, total=total))
    st.markdown(metadata.get("nota_evaluacion", metrics.get("nota_evaluacion", "")))
    ui.panel_end()

# --------------------------------------------------------------------------
# Hiperparámetros
# --------------------------------------------------------------------------
col3, col4 = st.columns(2)
with col3:
    ui.panel_start(i18n.t("panel_hparams_ae"), "")
    st.json(hparams.get("autoencoder", {}))
    ui.panel_end()
with col4:
    ui.panel_start(i18n.t("panel_hparams_dqn"), "")
    st.json(hparams.get("dqn_agent", {}))
    ui.panel_end()

# --------------------------------------------------------------------------
# Historial de reentrenamientos + trigger manual
# --------------------------------------------------------------------------
ui.panel_start(i18n.t("panel_retrain_history"), i18n.t("panel_retrain_history_sub"))
history = metadata.get("retrain_history", [])
if history:
    import pandas as pd
    st.dataframe(pd.DataFrame(history), use_container_width=True)
else:
    st.info(i18n.t("no_retrain_history"))

if st.button(i18n.t("btn_trigger_retrain")):
    try:
        result = api_client.trigger_retrain(api_url)
        st.success(i18n.t("retrain_queued").format(run_id=result['run_id'], status=result['status']))
    except Exception as e:
        st.error(i18n.t("retrain_error").format(error=e))
ui.panel_end()
