"""Simulation argument form — mirrors parser.add_common_sim_args."""
from __future__ import annotations

import streamlit as st


def render_sim_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Simulation")

        c1, c2 = st.columns(2)
        with c1:
            lpt_order = st.number_input("LPT order", min_value=1, max_value=3, value=defaults.get("lpt_order", 2), key=f"{prefix}lpt_order")
        with c2:
            nb_steps = st.number_input("Nb steps", min_value=1, value=defaults.get("nb_steps", 30), key=f"{prefix}nb_steps")

        c3, c4 = st.columns(2)
        with c3:
            t0 = st.number_input("t0", min_value=0.001, value=defaults.get("t0", 0.1), format="%.4f", key=f"{prefix}t0")
        with c4:
            t1 = st.number_input("t1", min_value=0.001, value=defaults.get("t1", 1.0), format="%.4f", key=f"{prefix}t1")

        interp = st.selectbox(
            "Interpolation",
            ["none", "onion", "telephoto"],
            index=["none", "onion", "telephoto"].index(defaults.get("interp", "none")),
            key=f"{prefix}interp",
        )
        scheme = st.selectbox(
            "Scheme",
            ["ngp", "bilinear", "rbf_neighbor"],
            index=["ngp", "bilinear", "rbf_neighbor"].index(defaults.get("scheme", "bilinear")),
            key=f"{prefix}scheme",
        )

        use_paint = st.checkbox("Set paint_nside", value=defaults.get("paint_nside") is not None, key=f"{prefix}use_paint_nside")
        paint_nside = None
        if use_paint:
            paint_nside = st.number_input("paint_nside", min_value=1, value=defaults.get("paint_nside", 64), key=f"{prefix}paint_nside")

        use_kernel = st.checkbox("Set kernel_width_arcmin", value=defaults.get("kernel_width_arcmin") is not None, key=f"{prefix}use_kernel_width")
        kernel_width_arcmin = None
        if use_kernel:
            kernel_width_arcmin = st.number_input("kernel_width_arcmin (arcmin)", min_value=0.0, value=defaults.get("kernel_width_arcmin", 1.0), format="%.4f", key=f"{prefix}kernel_width_arcmin")

        enable_x64 = st.checkbox("Enable x64", value=defaults.get("enable_x64", False), key=f"{prefix}enable_x64")

        return {
            "lpt_order": lpt_order,
            "nb_steps": nb_steps,
            "t0": t0,
            "t1": t1,
            "interp": interp,
            "scheme": scheme,
            "paint_nside": paint_nside,
            "kernel_width_arcmin": kernel_width_arcmin,
            "enable_x64": enable_x64,
        }
