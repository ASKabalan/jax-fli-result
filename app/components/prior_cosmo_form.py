"""Prior cosmology form — sampling checkboxes, prior bounds, fixed params, IC."""
from __future__ import annotations

import streamlit as st


def render_prior_cosmo_form(
    prefix: str = "",
    defaults: dict | None = None,
    show_ic: bool = True,
) -> dict:
    """Render the prior cosmology form.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.
    show_ic:
        When True, render the Initial Conditions section.

    Returns
    -------
    dict with keys:
        omega_c, sigma8, h, omega_b, w0, wa, n_s, omega_k, omega_nu,
        omega_c_fixed, sigma8_fixed, h_fixed,
        prior_omega_c, prior_sigma8, prior_h  (each [min, max] or None when fixed),
        sample  (derived list, e.g. ["cosmo", "ic"]),
        and when show_ic=True:
            ic_fixed, initial_condition, prior_ic_gaussian.
    """
    defaults = defaults or {}

    def _sampled_row(
        label: str, key: str, default_val: float, default_min: float, default_max: float
    ):
        """One sampled-parameter row: value | Fixed checkbox, then prior [min, max]."""
        val_col, fix_col = st.columns([2, 1])
        with val_col:
            val = st.number_input(
                label,
                value=float(defaults.get(key, default_val)),
                format="%.4f",
                key=f"{prefix}{key}_val",
            )
        with fix_col:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            fixed = st.checkbox(
                "Fixed",
                value=bool(defaults.get(f"{key}_fixed", False)),
                key=f"{prefix}{key}_fixed",
            )
        prior_min = prior_max = None
        if not fixed:
            pm_col, px_col = st.columns(2)
            with pm_col:
                prior_min = st.number_input(
                    "prior min",
                    value=float(defaults.get(f"prior_{key}_min", default_min)),
                    format="%.4f",
                    key=f"{prefix}{key}_prior_min",
                )
            with px_col:
                prior_max = st.number_input(
                    "prior max",
                    value=float(defaults.get(f"prior_{key}_max", default_max)),
                    format="%.4f",
                    key=f"{prefix}{key}_prior_max",
                )
        return val, fixed, prior_min, prior_max

    with st.container(border=True):
        st.subheader("Cosmology")
        st.caption("Uncheck **Fixed** to sample a parameter (sampled by default).")

        omega_c, omega_c_fixed, oc_min, oc_max = _sampled_row(
            "omega_c", "omega_c", 0.2589, 0.1, 0.5
        )
        sigma8, sigma8_fixed, s8_min, s8_max = _sampled_row(
            "sigma8", "sigma8", 0.8159, 0.6, 1.0
        )
        h_val, h_fixed, h_min, h_max = _sampled_row("h", "h", 0.6774, 0.5, 0.9)

        st.divider()
        st.markdown("**Fixed cosmological parameters**")

        h1c, h2c = st.columns(2)
        with h1c:
            omega_b = st.number_input(
                "omega_b",
                value=float(defaults.get("omega_b", 0.0486)),
                format="%.4f",
                key=f"{prefix}omega_b",
            )
        with h2c:
            n_s = st.number_input(
                "n_s",
                value=float(defaults.get("n_s", 0.9667)),
                format="%.4f",
                key=f"{prefix}n_s",
            )

        wa_col, w0_col = st.columns(2)
        with wa_col:
            wa = st.number_input(
                "wa",
                value=float(defaults.get("wa", 0.0)),
                format="%.4f",
                key=f"{prefix}wa",
            )
        with w0_col:
            w0 = st.number_input(
                "w0",
                value=float(defaults.get("w0", -1.0)),
                format="%.4f",
                key=f"{prefix}w0",
            )

        nu_col, k_col = st.columns(2)
        with nu_col:
            omega_nu = st.number_input(
                "omega_nu",
                value=float(defaults.get("omega_nu", 0.0)),
                format="%.4f",
                key=f"{prefix}omega_nu",
            )
        with k_col:
            omega_k = st.number_input(
                "omega_k",
                value=float(defaults.get("omega_k", 0.0)),
                format="%.4f",
                key=f"{prefix}omega_k",
            )

        # IC section
        ic_fixed = True
        initial_condition = None
        prior_ic_gaussian = [0.0, 1.0]

        if show_ic:
            st.divider()
            st.subheader("Initial Conditions")
            ic_fixed = st.checkbox(
                "Fixed (not sampled)",
                value=bool(defaults.get("ic_fixed", False)),
                key=f"{prefix}ic_fixed",
            )
            if ic_fixed:
                with st.expander("IC parquet path"):
                    ic_path = st.text_input(
                        "IC parquet path",
                        value=defaults.get("initial_condition", ""),
                        key=f"{prefix}ic_path",
                    )
                    initial_condition = ic_path.strip() if ic_path else None
            else:
                pg_col1, pg_col2 = st.columns(2)
                with pg_col1:
                    ic_prior_min = st.number_input(
                        "IC prior min",
                        value=float(defaults.get("prior_ic_min", 0.0)),
                        format="%.4f",
                        key=f"{prefix}ic_prior_min",
                    )
                with pg_col2:
                    ic_prior_max = st.number_input(
                        "IC prior max",
                        value=float(defaults.get("prior_ic_max", 1.0)),
                        format="%.4f",
                        key=f"{prefix}ic_prior_max",
                    )
                prior_ic_gaussian = [ic_prior_min, ic_prior_max]

    # Derive --sample list
    sample = []
    if not omega_c_fixed or not sigma8_fixed or not h_fixed:
        sample.append("cosmo")
    if show_ic and not ic_fixed:
        sample.append("ic")

    return {
        "omega_c": omega_c,
        "sigma8": sigma8,
        "h": h_val,
        "omega_b": omega_b,
        "w0": w0,
        "wa": wa,
        "n_s": n_s,
        "omega_k": omega_k,
        "omega_nu": omega_nu,
        "omega_c_fixed": omega_c_fixed,
        "sigma8_fixed": sigma8_fixed,
        "h_fixed": h_fixed,
        "prior_omega_c": [oc_min, oc_max] if not omega_c_fixed else None,
        "prior_sigma8": [s8_min, s8_max] if not sigma8_fixed else None,
        "prior_h": [h_min, h_max] if not h_fixed else None,
        "ic_fixed": ic_fixed,
        "initial_condition": initial_condition,
        "prior_ic_gaussian": prior_ic_gaussian,
        "sample": sample,
    }
