"""Miscellaneous page-specific forms.

Contains forms that are specific to a single page but are kept here to
maintain the architecture rule: components own forms, views call them.
"""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_list


def render_2pcf_observable_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Render the 2PCF observable settings form.

    Covers geometry, physics parameters, and MCMC settings for the
    2PCF Inference page (TASK 12).

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.

    Returns
    -------
    dict with keys:
        nside, flatsky_npix, field_size, lmax, f_sky, sigma_e,
        nonlinear_fn, num_warmup, num_samples, batch_count,
        sampler, backend, enable_x64.
    """
    defaults = defaults or {}

    nside = None
    flatsky_npix = None
    field_size = None

    with st.container(border=True):
        st.subheader("2PCF Observable Settings")

        st.markdown("**Geometry**")
        geom_mode = st.radio(
            "geometry",
            ["Spherical (nside)", "Flat sky"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}geom_mode",
        )

        if geom_mode == "Spherical (nside)":
            nside = st.number_input(
                "nside",
                min_value=1,
                value=int(defaults.get("nside", 64)),
                key=f"{prefix}nside",
            )
        else:
            fp1, fp2 = st.columns(2)
            with fp1:
                _fp_h = st.number_input(
                    "H (pixels)", min_value=1, value=512, key=f"{prefix}fp_h"
                )
            with fp2:
                _fp_w = st.number_input(
                    "W (pixels)", min_value=1, value=512, key=f"{prefix}fp_w"
                )
            flatsky_npix = [_fp_h, _fp_w]
            ff1, ff2 = st.columns(2)
            with ff1:
                _ff_h = st.number_input(
                    "H (deg)", min_value=1, value=10, key=f"{prefix}ff_h"
                )
            with ff2:
                _ff_w = st.number_input(
                    "W (deg)", min_value=1, value=10, key=f"{prefix}ff_w"
                )
            field_size = [_ff_h, _ff_w]

        lmax_col, fsky_col = st.columns(2)
        with lmax_col:
            lmax = st.number_input(
                "lmax",
                min_value=1,
                value=int(defaults.get("lmax", 2047)),
                key=f"{prefix}lmax",
            )
        with fsky_col:
            f_sky = st.number_input(
                "f_sky",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("f_sky", 1.0)),
                format="%.3f",
                key=f"{prefix}f_sky",
            )

        sigma_e_col, nl_col = st.columns(2)
        with sigma_e_col:
            sigma_e = st.number_input(
                "sigma_e",
                value=float(defaults.get("sigma_e", 0.26)),
                format="%.4f",
                key=f"{prefix}sigma_e",
            )
        with nl_col:
            nonlinear_fn = st.selectbox(
                "nonlinear_fn",
                ["halofit", "linear"],
                key=f"{prefix}nonlinear_fn",
            )

        st.markdown("**MCMC**")
        wm_col, ns_col = st.columns(2)
        with wm_col:
            num_warmup = st.number_input(
                "num_warmup",
                min_value=0,
                value=int(defaults.get("num_warmup", 100)),
                key=f"{prefix}num_warmup",
            )
        with ns_col:
            num_samples = st.number_input(
                "num_samples",
                min_value=1,
                value=int(defaults.get("num_samples", 500)),
                key=f"{prefix}num_samples",
            )

        batch_count = st.number_input(
            "batch_count",
            min_value=1,
            value=int(defaults.get("batch_count", 10)),
            key=f"{prefix}batch_count",
        )

        sm_col, be_col = st.columns(2)
        with sm_col:
            sampler = st.selectbox(
                "sampler",
                ["NUTS", "HMC", "MCLMC"],
                key=f"{prefix}sampler",
            )
        with be_col:
            backend = st.selectbox(
                "backend",
                ["numpyro", "blackjax"],
                index=1,
                key=f"{prefix}backend",
            )

        enable_x64 = st.checkbox(
            "enable_x64",
            value=bool(defaults.get("enable_x64", False)),
            key=f"{prefix}enable_x64",
        )

    return {
        "nside": nside,
        "flatsky_npix": flatsky_npix,
        "field_size": field_size,
        "lmax": lmax,
        "f_sky": f_sky,
        "sigma_e": sigma_e,
        "nonlinear_fn": nonlinear_fn,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "batch_count": batch_count,
        "sampler": sampler,
        "backend": backend,
        "enable_x64": enable_x64,
    }


def render_extract_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Render the extract settings form for the Extract page (TASK 13).

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.

    Returns
    -------
    dict with keys:
        source, input_dir, repo_id, config, truth_parquet, output_file,
        set_name, cosmo_keys, field_statistic, power_statistic, ddof, enable_x64.
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Extract Settings")

        source = st.radio(
            "Data source",
            ["Local directory", "HuggingFace repo"],
            key=f"{prefix}source",
        )

        input_dir = None
        repo_id = None
        config = None

        if source == "Local directory":
            input_dir = st.text_input(
                "input_dir",
                value=defaults.get("input_dir", "test_fli_samples"),
                key=f"{prefix}input_dir",
            )
        else:
            repo_id = st.text_input(
                "repo_id (e.g. 'ASKabalan/jax-fli-experiments')",
                key=f"{prefix}repo_id",
            )
            config = (
                render_dynamic_list("config", f"{prefix}config", [], cast_fn=str)
                or None
            )

        truth_parquet = st.text_input(
            "truth_parquet",
            value=defaults.get(
                "truth_parquet",
                "test_fli_samples/chain_0/samples/samples_0.parquet",
            ),
            key=f"{prefix}truth",
        )
        output_file = st.text_input(
            "output_file",
            value=defaults.get("output_file", "results/extracts/extract.parquet"),
            key=f"{prefix}output",
        )
        set_name = st.text_input(
            "set_name",
            value=defaults.get("set_name", "my_extract"),
            key=f"{prefix}set_name",
        )

        cosmo_keys = render_dynamic_list(
            "cosmo_keys",
            f"{prefix}cosmo_keys",
            defaults.get("cosmo_keys", ["Omega_c", "sigma8"]),
            cast_fn=str,
        )

        fs_col, ps_col = st.columns(2)
        with fs_col:
            field_statistic = st.checkbox(
                "field_statistic",
                value=bool(defaults.get("field_statistic", True)),
                key=f"{prefix}field_stat",
            )
        with ps_col:
            power_statistic = st.checkbox(
                "power_statistic",
                value=bool(defaults.get("power_statistic", True)),
                key=f"{prefix}power_stat",
            )

        ddof = st.number_input(
            "ddof",
            min_value=0,
            value=int(defaults.get("ddof", 0)),
            key=f"{prefix}ddof",
        )
        enable_x64 = st.checkbox(
            "enable_x64",
            value=bool(defaults.get("enable_x64", False)),
            key=f"{prefix}enable_x64",
        )

    return {
        "source": source,
        "input_dir": input_dir,
        "repo_id": repo_id,
        "config": config,
        "truth_parquet": truth_parquet,
        "output_file": output_file,
        "set_name": set_name,
        "cosmo_keys": cosmo_keys,
        "field_statistic": field_statistic,
        "power_statistic": power_statistic,
        "ddof": ddof,
        "enable_x64": enable_x64,
    }
