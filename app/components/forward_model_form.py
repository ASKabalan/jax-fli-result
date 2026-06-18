"""Forward-model likelihood form — mirrors parser.add_forward_model_args.

Used by the Full Field Inference and Samples pages (fli-infer / fli-samples): a survey footprint
mask, the inflated likelihood sigma on unobserved pixels, and whether to log the lightcone.
"""
from __future__ import annotations

import streamlit as st

_MASK_NONE = "none"
_MASK_DESY3 = "des_y3"
_MASK_FILE = "file (path)"


def render_forward_model_form(defaults: dict | None = None, prefix: str = "") -> dict:
    """Render the full-field forward-model likelihood settings.

    Returns a dict with keys: ``mask`` (None / 'des_y3' / path), ``sigma_unobserved``,
    ``log_lightcone``.
    """
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Forward Model")

        mode = st.selectbox(
            "mask",
            [_MASK_NONE, _MASK_DESY3, _MASK_FILE],
            key=f"{prefix}fm_mask_mode",
            help="Survey footprint for the likelihood; pixels outside it get sigma_unobserved.",
        )
        mask = None
        if mode == _MASK_DESY3:
            mask = "des_y3"
        elif mode == _MASK_FILE:
            mask = (
                st.text_input(
                    "mask path",
                    value=str(defaults.get("mask") or ""),
                    key=f"{prefix}fm_mask_path",
                    help="Path to a HEALPix map (.npy / .npz / .fits).",
                ).strip()
                or None
            )

        sigma_unobserved = st.number_input(
            "sigma_unobserved",
            min_value=0.0,
            value=float(defaults.get("sigma_unobserved", 1e6)),
            format="%.6g",
            key=f"{prefix}fm_sigma_unobserved",
            help="Likelihood sigma applied on pixels outside the mask.",
        )
        log_lightcone = st.checkbox(
            "log_lightcone",
            value=bool(defaults.get("log_lightcone", False)),
            key=f"{prefix}fm_log_lightcone",
            help="Record the lightcone as a deterministic site in the trace.",
        )

    return {
        "mask": mask,
        "sigma_unobserved": sigma_unobserved,
        "log_lightcone": log_lightcone,
    }
