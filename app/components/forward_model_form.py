"""Forward-model likelihood form — mirrors parser.add_forward_model_args.

Used by the Full Field Inference and Samples pages (fli-infer / fli-samples): the lensing
observable (convergence vs shear — a forward-model concern), a survey footprint mask, the inflated
likelihood sigma on unobserved pixels, the lightcone-logging toggle, the observer-visibility
apodization scale, and the map2alm method.
"""
from __future__ import annotations

import streamlit as st

_MASK_NONE = "none"
_MASK_DESY3 = "des_y3"
_MASK_FILE = "file (path)"
_LENSING_OUTPUTS = ["convergence", "shear", "reduced_shear"]
_MAP2ALM_METHODS = ["jax", "jax_cuda"]


def render_forward_model_form(defaults: dict | None = None, prefix: str = "") -> dict:
    """Render the full-field forward-model likelihood settings.

    Returns a dict with keys: ``lensing_output`` ('convergence' / 'shear' / 'reduced_shear'),
    ``mask`` (None / 'des_y3' / path), ``sigma_unobserved``, ``log_lightcone``,
    ``apodization_scale_deg``, ``map2alm_method``.
    """
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Forward Model")

        lensing_output = st.selectbox(
            "lensing_output",
            _LENSING_OUTPUTS,
            index=_LENSING_OUTPUTS.index(defaults.get("lensing_output", "convergence")),
            key=f"{prefix}fm_lensing_output",
            help="Observable the forward model produces: convergence (kappa) or spin-2 shear / "
            "reduced_shear via Kaiser-Squires.",
        )

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

        ap_col, m2a_col = st.columns(2)
        with ap_col:
            apodization_scale_deg = st.number_input(
                "apodization_scale_deg",
                min_value=0.0,
                value=float(defaults.get("apodization_scale_deg", 1.0)),
                format="%.2f",
                key=f"{prefix}fm_apodization_scale_deg",
                help="C2 apodization scale (deg) for the off-center observer visibility mask.",
            )
        with m2a_col:
            map2alm_method = st.selectbox(
                "map2alm_method",
                _MAP2ALM_METHODS,
                index=_MAP2ALM_METHODS.index(defaults.get("map2alm_method", "jax")),
                key=f"{prefix}fm_map2alm_method",
                help="Method for the map→alm conversion.",
            )

    return {
        "lensing_output": lensing_output,
        "mask": mask,
        "sigma_unobserved": sigma_unobserved,
        "log_lightcone": log_lightcone,
        "apodization_scale_deg": apodization_scale_deg,
        "map2alm_method": map2alm_method,
    }
