"""Cosmology argument form — mirrors parser.add_cosmo_args."""
from __future__ import annotations

import streamlit as st


def render_cosmo_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Cosmology")

        c1, c2 = st.columns(2)
        with c1:
            h = st.number_input("h", value=defaults.get("h", 0.6774), format="%.4f", key=f"{prefix}h")
        with c2:
            omega_b = st.number_input("omega_b", value=defaults.get("omega_b", 0.0486), format="%.4f", key=f"{prefix}omega_b")

        c3, c4 = st.columns(2)
        with c3:
            omega_k = st.number_input("omega_k", value=defaults.get("omega_k", 0.0), format="%.4f", key=f"{prefix}omega_k")
        with c4:
            omega_nu = st.number_input("omega_nu", value=defaults.get("omega_nu", 0.0), format="%.4f", key=f"{prefix}omega_nu")

        c5, c6 = st.columns(2)
        with c5:
            w0 = st.number_input("w0", value=defaults.get("w0", -1.0), format="%.4f", key=f"{prefix}w0")
        with c6:
            wa = st.number_input("wa", value=defaults.get("wa", 0.0), format="%.4f", key=f"{prefix}wa")

        n_s = st.number_input("n_s", value=defaults.get("n_s", 0.9667), format="%.4f", key=f"{prefix}n_s")

        return {
            "h": h,
            "omega_b": omega_b,
            "omega_k": omega_k,
            "omega_nu": omega_nu,
            "w0": w0,
            "wa": wa,
            "n_s": n_s,
        }
