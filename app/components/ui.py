"""
ui.py — componentes visuales reutilizables entre páginas.

Mantiene el HTML/CSS repetitivo fuera de las páginas, para que cada página
se lea como lógica de negocio + layout, no como sopa de <div>s.
"""

from pathlib import Path

import streamlit as st
from components import i18n

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

SEVERITY_LABEL = {"ok": "OPERANDO", "warning": "ATENCIÓN", "critical": "CRÍTICO", "offline": "SIN DATOS"}


def load_css():
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<div class="pm-eyebrow">{text}</div>', unsafe_allow_html=True)


def status_pill(severity: str, label: str = None) -> str:
    if not label:
        label = i18n.t(severity.upper(), severity.upper())
    else:
        label = i18n.t(label, label)
    return f'<span class="pm-pill {severity}"><span class="pm-pill-dot"></span>{label}</span>'


def kpi_card(label: str, value: str, sub: str = "", accent: str = "var(--accent-signal)") -> str:
    return f'<div class="pm-kpi-card" style="--kpi-accent:{accent};"><div class="pm-kpi-label">{label}</div><div class="pm-kpi-value">{value}</div><div class="pm-kpi-sub">{sub}</div></div>'


def kpi_grid(cards_html: list):
    st.markdown(f'<div class="pm-kpi-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def panel_start(title: str, sub: str = ""):
    st.markdown(
        f'<div class="pm-panel"><div class="pm-panel-title">{title}</div><div class="pm-panel-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def alert_row(ts: str, machine_id: str, severity: str, action_label: str, reconstruction_error: float) -> str:
    row_color = {"warning": "var(--status-warning)", "critical": "var(--status-critical)"}
    accent = row_color.get(severity, '#7C8AA3')
    action_label = i18n.t(action_label, action_label)
    pill = status_pill(severity, action_label)
    return (
        f'<div class="pm-alert-row" style="--row-accent:{accent};">'
        f'<span class="pm-alert-time">{ts}</span>'
        f'<span class="pm-alert-machine">{machine_id}</span>'
        f'{pill}'
        f'<span style="color:var(--text-secondary); font-family:var(--font-mono); font-size:0.78rem; margin-left:auto;">'
        f'error = {reconstruction_error:.4f}'
        f'</span>'
        f'</div>'
    )


def api_status_banner(health: dict, api_url: str):
    if health is None:
        html = (
            f'<div class="pm-panel" style="border-color: var(--status-critical);">'
            f'{status_pill("critical", i18n.t("API NO DISPONIBLE"))} '
            f'<span style="color:var(--text-secondary); margin-left:10px; font-size:0.85rem;">'
            f'{i18n.t("No se pudo conectar a")} <code>{api_url}</code>. {i18n.t("¿Está corriendo")} <code>uvicorn main:app</code>?'
            f'</span></div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        return False
    ok = health.get("models_loaded", False)
    sev = "ok" if ok else "warning"
    label = i18n.t("SISTEMA OPERATIVO") if ok else i18n.t("MODELOS NO CARGADOS")
    html = (
        f'<div class="pm-panel" style="padding:12px 20px; display:flex; align-items:center; gap:16px;">'
        f'{status_pill(sev, label)} '
        f'<span style="color:var(--text-secondary); font-size:0.8rem; font-family:var(--font-mono);">'
        f'{i18n.t("uptime")}: {health.get("uptime_seconds", 0)}s · {i18n.t("API")}: {api_url}'
        f'</span></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    return True
