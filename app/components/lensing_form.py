"""Lensing argument form — mirrors parser.add_lensing_args."""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_list


def render_lensing_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Lensing")

        # Determine initial mode from existing default
        default_nz = defaults.get("nz_shear", ["s3"])
        if isinstance(default_nz, str):
            default_nz = [default_nz]
        is_s3_default = len(default_nz) == 1 and str(default_nz[0]).lower().startswith("s3")
        default_mode = "s3 preset" if is_s3_default else "custom z values"

        mode = st.radio(
            "nz_shear mode",
            ["s3 preset", "custom z values"],
            index=0 if default_mode == "s3 preset" else 1,
            horizontal=True,
            key=f"{prefix}nz_shear_mode",
        )

        if mode == "s3 preset":
            s3_val = st.text_input(
                "nz_shear (s3 notation)",
                value=str(default_nz[0]) if is_s3_default else "s3",
                help="e.g. s3, s3[0], s3[1:3], s3[:2], s3[::2]",
                key=f"{prefix}nz_shear_s3",
            )
            nz_shear = [s3_val]
        else:
            raw = render_dynamic_list(
                "nz_shear z-values",
                f"{prefix}nz_shear_custom",
                [] if is_s3_default else [float(v) for v in default_nz],
                cast_fn=float,
            )
            nz_shear = [str(v) for v in raw] if raw else ["s3"]

        c1, c2 = st.columns(2)
        with c1:
            min_z = st.number_input("min_z", min_value=0.0, value=defaults.get("min_z", 0.01), format="%.4f", key=f"{prefix}min_z")
        with c2:
            max_z = st.number_input("max_z", min_value=0.0, value=defaults.get("max_z", 1.5), format="%.4f", key=f"{prefix}max_z")

        n_integrate = st.number_input("n_integrate", min_value=1, value=defaults.get("n_integrate", 32), key=f"{prefix}n_integrate")

        return {
            "nz_shear": nz_shear,
            "min_z": min_z,
            "max_z": max_z,
            "n_integrate": n_integrate,
        }
