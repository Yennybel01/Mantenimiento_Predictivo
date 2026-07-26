"""
api_client.py — Cliente HTTP hacia el backend FastAPI.

Todas las páginas del dashboard pasan por aquí para hablar con la API.
Ninguna página debería usar `requests` directamente ni conocer los
artefactos de ML: si mañana la API cambia de host, de puerto, o incluso de
framework, solo se toca este archivo.
"""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("PM_API_URL", "http://localhost:8000")


def get_api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def _url(path: str) -> str:
    return f"{get_api_url().rstrip('/')}{path}"


@st.cache_data(ttl=5)
def check_health(api_url: str):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=15)
def get_metadata(api_url: str):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/model/metadata", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(records: list, api_url: str):
    r = requests.post(f"{api_url.rstrip('/')}/predict", json={"records": records}, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


@st.cache_data(ttl=5)
def get_alerts(api_url: str, limit: int = 50, severity: str = None):
    params = {"limit": limit}
    if severity:
        params["severity"] = severity
    r = requests.get(f"{api_url.rstrip('/')}/alerts", params=params, timeout=5)
    r.raise_for_status()
    return r.json()["alerts"]


def trigger_retrain(api_url: str):
    r = requests.post(f"{api_url.rstrip('/')}/retrain", timeout=5)
    r.raise_for_status()
    return r.json()
