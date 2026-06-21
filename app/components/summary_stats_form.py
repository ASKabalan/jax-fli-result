"""Summary-statistics argument forms — mirror the add_summary_stats_*_args groups in parser.py."""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import parse_csv_list, render_dynamic_list

_PAINT_ORDERS = ["ngp", "cic", "tsc", "pcs"]


def render_summary_stats_scan_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_scan_args: folder scan and filter settings."""
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


def render_summary_stats_flat_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_flat_args: flat-sky angular Cl bin edges."""
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


def render_summary_stats_spherical_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_spherical_args: spherical HEALPix Cl settings."""
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


def render_summary_stats_density_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_density_args: 3D P(k) bin configuration + window corrections."""
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

        co_col, sn_col = st.columns(2)
        with co_col:
            _co = st.selectbox(
                "compensate_order",
                ["off", *_PAINT_ORDERS],
                key=f"{prefix}compensate_order",
                help="Deconvolve the mass-assignment window of this order from P(k).",
            )
            compensate_order = None if _co == "off" else _co
        with sn_col:
            _sn = st.selectbox(
                "shotnoise_order",
                ["off", *_PAINT_ORDERS],
                key=f"{prefix}shotnoise_order",
                help="Subtract aliased shot noise for this assignment order (auto-spectrum).",
            )
            shotnoise_order = None if _sn == "off" else _sn
    return {
        "kedges": kedges,
        "dk": dk,
        "kmax": kmax,
        "multipoles": multipoles,
        "los": [los_x, los_y, los_z],
        "compensate_order": compensate_order,
        "shotnoise_order": shotnoise_order,
    }


def render_summary_stats_mask_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_mask_args: spherical footprint mask + apodization.

    The mask restricts the observed footprint of spherical maps before the statistic is computed.
    """
    defaults = defaults or {}
    _MODES = ["infer_from_observer_position", "none", "des_y3", "file (path)"]
    with st.container(border=True):
        st.subheader("Mask (spherical maps)")
        _default = defaults.get("mask", "infer_from_observer_position")
        _mode_default = _default if _default in _MODES else "file (path)"
        mode = st.selectbox(
            "mask",
            _MODES,
            index=_MODES.index(_mode_default),
            key=f"{prefix}mask_mode",
            help="Footprint for spherical stats; 'infer_from_observer_position' is a no-op for a "
            "centered observer.",
        )
        if mode == "file (path)":
            mask = (
                st.text_input(
                    "mask path",
                    value=_default if _default not in _MODES else "",
                    key=f"{prefix}mask_path",
                    help="Path to a HEALPix map (.npy / .npz / .fits).",
                ).strip()
                or "none"
            )
        else:
            mask = mode

        apodization_scale_deg = st.number_input(
            "apodization_scale_deg",
            min_value=0.0,
            value=float(defaults.get("apodization_scale_deg", 1.0)),
            format="%.2f",
            key=f"{prefix}apodization_scale_deg",
            help="C2 apodization scale (deg) applied to the mask.",
        )

        override_obs = st.checkbox(
            "override observer position",
            value=defaults.get("observer_position") is not None,
            key=f"{prefix}override_obs",
            help="By default the observer position is read from the field metadata.",
        )
        observer_position = None
        if override_obs:
            _obs = defaults.get("observer_position") or [0.5, 0.5, 0.5]
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                ox = st.number_input(
                    "OX", value=float(_obs[0]), format="%.2f", key=f"{prefix}obs_x"
                )
            with oc2:
                oy = st.number_input(
                    "OY", value=float(_obs[1]), format="%.2f", key=f"{prefix}obs_y"
                )
            with oc3:
                oz = st.number_input(
                    "OZ", value=float(_obs[2]), format="%.2f", key=f"{prefix}obs_z"
                )
            observer_position = [ox, oy, oz]
    return {
        "mask": mask,
        "apodization_scale_deg": apodization_scale_deg,
        "observer_position": observer_position,
    }


def render_summary_stats_common_form(
    prefix: str = "", defaults: dict | None = None
) -> dict:
    """Mirror add_summary_stats_common_args + add_common_args: batch size and x64 precision."""
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
