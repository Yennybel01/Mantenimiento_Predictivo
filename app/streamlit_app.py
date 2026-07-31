import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import api_client, ui, i18n

st.set_page_config(
    page_title="Predictive Maintenance — Control Room",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Sidebar: configuración de conexión a la API (se ejecuta en cada renderizado)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="pm-eyebrow">{i18n.t("eyebrow_sidebar")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {i18n.t('title_sidebar')}")
    st.caption(i18n.t("caption_sidebar"))
    st.markdown("---")
    
    # Selector de idioma
    current_lang = i18n.get_lang()
    options = {"es": "Español", "en": "English"}
    lang_keys = list(options.keys())
    idx = lang_keys.index(current_lang) if current_lang in lang_keys else 0
    
    selected_label = st.selectbox(
        i18n.t("Language / Idioma"),
        options=list(options.values()),
        index=idx,
        key="lang_selector_widget"
    )
    selected_lang = [k for k, v in options.items() if v == selected_label][0]
    if selected_lang != current_lang:
        st.session_state["lang"] = selected_lang
        st.rerun()

    st.markdown("---")
    api_url = st.text_input(i18n.t("api_url_label"), value=api_client.get_api_url())
    st.session_state["api_url"] = api_url
    st.markdown("---")
    st.caption(i18n.t("dashboard_desc"))

ui.load_css()

# --------------------------------------------------------------------------
# Función para renderizar la página principal
# --------------------------------------------------------------------------
def show_overview():
    health = api_client.check_health(api_url)

    # Header
    ui.eyebrow(i18n.t("eyebrow_home"))
    st.markdown(f"# {i18n.t('title_home')}")
    st.caption(i18n.t("subtitle_home"))

    is_online = ui.api_status_banner(health, api_url)

    if not is_online:
        st.info(i18n.t("backend_start_info"))
        st.stop()

    # KPIs
    metadata = api_client.get_metadata(api_url) or {}
    alerts = api_client.get_alerts(api_url, limit=200) or []
    n_warning = len([a for a in alerts if a["severity"] == "warning"])
    n_critical = len([a for a in alerts if a["severity"] == "critical"])

    cards = [
        ui.kpi_card(i18n.t("kpi_version"), metadata.get("version", "—"), metadata.get("status", ""), "var(--accent-signal)"),
        ui.kpi_card(i18n.t("kpi_threshold"), f"{metadata.get('threshold', 0):.4f}", i18n.t("error de reconstrucción"), "var(--accent-signal)"),
        ui.kpi_card(i18n.t("kpi_alerts_warning"), str(n_warning), i18n.t("Mantenimiento preventivo"), "var(--status-warning)"),
        ui.kpi_card(i18n.t("kpi_alerts_critical"), str(n_critical), i18n.t("Parada de emergencia"), "var(--status-critical)"),
    ]
    ui.kpi_grid(cards)

    # Navegación rápida
    ui.panel_start(i18n.t("panel_modules"), i18n.t("panel_modules_sub"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.page_link("pages/1_📡_Monitoreo.py", label=i18n.t("link_monitoreo"), help=i18n.t("link_monitoreo_help"))
    with c2:
        st.page_link("pages/2_📊_Análisis.py", label=i18n.t("link_analisis"), help=i18n.t("link_analisis_help"))
    with c3:
        st.page_link("pages/3_🔧_Predicciones.py", label=i18n.t("link_predicciones"), help=i18n.t("link_predicciones_help"))
    with c4:
        st.page_link("pages/4_🚨_Alertas.py", label=i18n.t("link_alertas"), help=i18n.t("link_alertas_help"))
    ui.panel_end()

    # Resumen de arquitectura
    ui.panel_start(i18n.t("panel_arch"), i18n.t("panel_arch_sub"))
    st.markdown(
        f"""```
{i18n.t("arch_desc")}
```
**{i18n.t("api_contract")}** (`/docs` {i18n.t("interactive_ref")}):
`GET /health` · `GET /model/metadata` · `POST /predict` · `GET /alerts` · `POST /retrain`"""
    )
    ui.panel_end()


# --------------------------------------------------------------------------
# Definición y ejecución de la navegación
# --------------------------------------------------------------------------
overview_page = st.Page(show_overview, title=i18n.t("title_home"), icon="⚙️", default=True)
monitoreo_page = st.Page("pages/1_📡_Monitoreo.py", title=i18n.t("link_monitoreo"), icon="📡")
analisis_page = st.Page("pages/2_📊_Análisis.py", title=i18n.t("link_analisis"), icon="📊")
predicciones_page = st.Page("pages/3_🔧_Predicciones.py", title=i18n.t("link_predicciones"), icon="🔧")
alertas_page = st.Page("pages/4_🚨_Alertas.py", title=i18n.t("link_alertas"), icon="🚨")

pg = st.navigation([overview_page, monitoreo_page, analisis_page, predicciones_page, alertas_page])
pg.run()
