"""Spectra argument forms — mirror the five add_spectra_*_args groups in parser.py."""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import parse_csv_list, render_dynamic_list


def render_spectra_scan_form(prefix: str = "", defaults: dict | None = None) -> dict:
    """Mirror add_spectra_scan_args: folder scan and filter settings."""
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Scan")
        folder = st.text_input(
            "folder",
            value=defaults.get("folder", "results/cosmology_runs"),
            key=f"{prefix}folder",
            help="Root folder to scan for parquet files.",
        )
        regex = st.text_input(
            "regex",
            value=defaults.get("regex", r".*\.parquet$"),
            key=f"{prefix}regex",
            help="Regex pattern to filter filenames (default: all .parquet files).",
        )
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            recursive = st.checkbox(
                "recursive",
                value=defaults.get("recursive", False),
                key=f"{prefix}recursive",
            )
        with rc2:
            force_regen = st.checkbox(
                "force_regen",
                value=defaults.get("force_regen", False),
                key=f"{prefix}force_regen",
            )
        with rc3:
            normalization = st.selectbox(
                "normalization",
                ["global", "per_plane"],
                index=["global", "per_plane"].index(
                    defaults.get("normalization", "global")
                ),
                key=f"{prefix}normalization",
            )
    return {
        "folder": folder,
        "regex": regex,
        "recursive": recursive,
        "force_regen": force_regen,
        "normalization": normalization,
    }


def render_spectra_flat_form(prefix: str = "", defaults: dict | None = None) -> dict:
    """Mirror add_spectra_flat_args: flat-sky angular Cl bin edges."""
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Flat-sky C_ell")
        raw_ell = st.text_input(
            "ell_edges (comma-separated)",
            value=", ".join(str(v) for v in (defaults.get("ell_edges") or [])),
            key=f"{prefix}ell_edges_csv",
            help="e.g. 10, 50, 100, 500. Leave empty for auto.",
        )
        ell_edges = parse_csv_list(raw_ell, float) or None
    return {"ell_edges": ell_edges}


def render_spectra_spherical_form(prefix: str = "", defaults: dict | None = None) -> dict:
    """Mirror add_spectra_spherical_args: spherical HEALPix Cl settings."""
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Spherical C_ell")
        use_lmax = st.checkbox(
            "Override lmax",
            value=defaults.get("lmax") is not None,
            key=f"{prefix}use_lmax",
        )
        lmax = None
        if use_lmax:
            lmax = st.number_input(
                "lmax",
                min_value=1,
                value=defaults.get("lmax", 511),
                key=f"{prefix}lmax",
                help="Default: 3*nside-1",
            )
        method = st.selectbox(
            "SHT method",
            ["healpy", "jax"],
            index=["healpy", "jax"].index(defaults.get("method", "healpy")),
            key=f"{prefix}method",
        )
    return {"lmax": lmax, "method": method}


def render_spectra_density_form(prefix: str = "", defaults: dict | None = None) -> dict:
    """Mirror add_spectra_density_args: 3D P(k) bin configuration."""
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("3D P(k)")
        k_mode = st.radio(
            "k bins",
            ["Auto", "Custom Edges", "Custom dk"],
            horizontal=True,
            key=f"{prefix}k_mode",
            help="Auto: CLI defaults. Custom Edges: explicit bin edges. Custom dk: uniform bins.",
        )
        kedges = dk = kmax = None
        if k_mode == "Custom Edges":
            raw = st.text_input(
                "kedges (comma-separated)",
                value=", ".join(str(v) for v in (defaults.get("kedges") or [])),
                key=f"{prefix}kedges_csv",
                help="e.g. 1e-3, 5e-3, 1e-2, 1e-1",
            )
            kedges = parse_csv_list(raw, float) or None
        elif k_mode == "Custom dk":
            dk_col, kmax_col = st.columns(2)
            with dk_col:
                dk = st.number_input(
                    "dk",
                    value=defaults.get("dk", 0.01),
                    format="%.4f",
                    key=f"{prefix}dk",
                )
            with kmax_col:
                kmax = st.number_input(
                    "kmax",
                    value=defaults.get("kmax", 1.0),
                    format="%.4f",
                    key=f"{prefix}kmax",
                )

        _default_multipoles = [str(v) for v in (defaults.get("multipoles") or [0])]
        multipoles = render_dynamic_list(
            "multipoles",
            f"{prefix}multipoles",
            _default_multipoles,
            cast_fn=int,
        ) or [0]

        _los = defaults.get("los") or [0.0, 0.0, 1.0]
        lo1, lo2, lo3 = st.columns(3)
        with lo1:
            los_x = st.number_input(
                "LOS x", value=float(_los[0]), format="%.2f", key=f"{prefix}los_x"
            )
        with lo2:
            los_y = st.number_input(
                "LOS y", value=float(_los[1]), format="%.2f", key=f"{prefix}los_y"
            )
        with lo3:
            los_z = st.number_input(
                "LOS z", value=float(_los[2]), format="%.2f", key=f"{prefix}los_z"
            )
    return {
        "kedges": kedges,
        "dk": dk,
        "kmax": kmax,
        "multipoles": multipoles,
        "los": [los_x, los_y, los_z],
    }


def render_spectra_common_form(prefix: str = "", defaults: dict | None = None) -> dict:
    """Mirror add_spectra_common_args: shared batch size and precision settings."""
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Common")
        bs_col, x64_col = st.columns(2)
        with bs_col:
            use_batch = st.checkbox(
                "Set batch size",
                value=defaults.get("batch_size") is not None,
                key=f"{prefix}use_batch",
            )
            batch_size = None
            if use_batch:
                batch_size = st.number_input(
                    "batch_size",
                    min_value=1,
                    value=defaults.get("batch_size", 4),
                    key=f"{prefix}batch_size",
                )
        with x64_col:
            enable_x64 = st.checkbox(
                "enable_x64",
                value=defaults.get("enable_x64", False),
                key=f"{prefix}enable_x64",
            )
    return {"batch_size": batch_size, "enable_x64": enable_x64}
