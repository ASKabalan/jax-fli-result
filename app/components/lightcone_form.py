"""Lightcone / shell argument form — mirrors parser.add_lightcone_args."""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_list


def render_lightcone_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Lightcone")

        c1, c2 = st.columns(2)
        with c1:
            nb_shells = st.number_input(
                "nb_shells",
                min_value=1,
                value=defaults.get("nb_shells", 10),
                key=f"{prefix}nb_shells",
            )
        with c2:
            halo_multiplier = st.number_input(
                "halo_multiplier",
                min_value=0.0,
                value=defaults.get("halo_multiplier", 0.5),
                step=0.1,
                format="%.2f",
                key=f"{prefix}halo_multiplier",
            )

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        obs_default = defaults.get("observer_position", [0.5, 0.5, 0.5])
        with oc1:
            ox = st.number_input(
                "OX", value=obs_default[0], format="%.2f", key=f"{prefix}obs_x"
            )
        with oc2:
            oy = st.number_input(
                "OY", value=obs_default[1], format="%.2f", key=f"{prefix}obs_y"
            )
        with oc3:
            oz = st.number_input(
                "OZ", value=obs_default[2], format="%.2f", key=f"{prefix}obs_z"
            )

        use_ts = st.checkbox(
            "Custom shell scale factors (ts)",
            value=defaults.get("ts") is not None,
            key=f"{prefix}use_ts",
        )
        ts = None
        ts_near = None
        ts_far = None
        if use_ts:
            ts = render_dynamic_list(
                "ts", f"{prefix}ts", defaults.get("ts") or [], cast_fn=float
            )
            ts_near = render_dynamic_list(
                "ts_near",
                f"{prefix}ts_near",
                defaults.get("ts_near") or [],
                cast_fn=float,
            )
            ts_far = render_dynamic_list(
                "ts_far", f"{prefix}ts_far", defaults.get("ts_far") or [], cast_fn=float
            )
            ts = ts or None
            ts_near = ts_near or None
            ts_far = ts_far or None

        drift_on_lightcone = st.checkbox(
            "drift_on_lightcone",
            value=defaults.get("drift_on_lightcone", False),
            key=f"{prefix}drift_on_lightcone",
        )
        equal_vol = st.checkbox(
            "equal_vol",
            value=defaults.get("equal_vol", False),
            key=f"{prefix}equal_vol",
            help="Equal-volume shell partitioning (fli-infer/fli-samples)",
        )
        min_width = st.number_input(
            "min_width",
            min_value=0.0,
            value=defaults.get("min_width", 50.0),
            format="%.1f",
            key=f"{prefix}min_width",
        )

        return {
            "nb_shells": nb_shells,
            "halo_multiplier": halo_multiplier,
            "observer_position": [ox, oy, oz],
            "ts": ts,
            "ts_near": ts_near,
            "ts_far": ts_far,
            "drift_on_lightcone": drift_on_lightcone,
            "equal_vol": equal_vol,
            "min_width": min_width,
        }
