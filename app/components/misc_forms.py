"""Miscellaneous page-specific forms.

Contains forms that are specific to a single page but are kept here to
maintain the architecture rule: components own forms, views call them.
"""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_list


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
        truth_parquet, output_file, set_name, cosmo_keys, field_statistic,
        power_statistic, ddof, enable_x64. The source (--input / --repo /
        --data-files, multi-pattern) is rendered separately by render_source_form.
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Extract Settings")

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
        "truth_parquet": truth_parquet,
        "output_file": output_file,
        "set_name": set_name,
        "cosmo_keys": cosmo_keys,
        "field_statistic": field_statistic,
        "power_statistic": power_statistic,
        "ddof": ddof,
        "enable_x64": enable_x64,
    }
