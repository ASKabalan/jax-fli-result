"""Lensing post-processing form — mirrors parser.add_lensing_postproc_args."""
from __future__ import annotations

import streamlit as st

_NORMALIZATIONS = ["global", "per_plane"]


def render_lensing_postproc_form(
    defaults: dict | None = None,
    prefix: str = "",
    *,
    output_default: str = ".",
) -> dict:
    """Output dir + density→κ knobs for fli-born-rt / fli-dorian-rt.

    Mirrors ``jax_fli.scripts.parser.add_lensing_postproc_args``: where the κ parquet is written,
    the ud_grade downsample nside, and the density→δ overdensity normalization.
    """
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Lensing post-processing")

        output_path = st.text_input(
            "output",
            value=defaults.get("output", output_default),
            key=f"{prefix}output",
        )

        c1, c2 = st.columns(2)
        with c1:
            nside_val = st.number_input(
                "nside (0 = native)",
                min_value=0,
                value=defaults.get("nside", 0) or 0,
                key=f"{prefix}nside",
                help="Downsample the density to this HEALPix nside before lensing. 0 keeps native.",
            )
        with c2:
            normalization = st.selectbox(
                "normalization",
                _NORMALIZATIONS,
                index=_NORMALIZATIONS.index(defaults.get("normalization", "global")),
                key=f"{prefix}normalization",
                help="Overdensity normalization for the density→δ conversion.",
            )

        return {
            "output": output_path,
            "nside": int(nside_val) if nside_val else None,
            "normalization": normalization,
        }
