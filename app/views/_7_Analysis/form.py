"""Analysis page UI: file loading, field map visualization, and spectra routing.

This module owns all session-state management and top-level UI scaffolding.
Actual analysis logic is delegated to the analysis-specific modules.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .utils import (
    _DENSITY_3D,
    _FLAT_TYPES,
    _PARTICLE_TYPE,
    _SPECTRA_TYPES,
    _SPHERICAL_TYPES,
    _fig_to_png,
    _plt_lock,
)

# ---------------------------------------------------------------------------
# Session state constants
# ---------------------------------------------------------------------------

CATALOGS_KEY = "analysis_catalogs"


def _init_session_state():
    if CATALOGS_KEY not in st.session_state:
        st.session_state[CATALOGS_KEY] = []
    if "analysis_nz_point_sources" not in st.session_state:
        st.session_state["analysis_nz_point_sources"] = [0.01]


# ---------------------------------------------------------------------------
# Session state callbacks
# ---------------------------------------------------------------------------


def _load_catalog(path: str):
    import glob as _glob

    import jax_fli as jfli

    path = path.strip()
    if not path:
        return

    # Glob pattern (contains *, ?, or [)
    if any(c in path for c in ("*", "?", "[")):
        matched = sorted(_glob.glob(path, recursive=True))
        if not matched:
            st.error(f"No files matched pattern: {path}")
            return
        for pf in matched:
            if Path(pf).is_file():
                _load_catalog(pf)
        return

    p = Path(path)
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        if not parquet_files:
            st.error(f"No .parquet files found under {path}")
            return
        for pf in parquet_files:
            _load_catalog(str(pf))
        return

    for entry in st.session_state[CATALOGS_KEY]:
        if entry["path"] == path:
            st.toast(f"Already loaded: {Path(path).name}")
            return
    try:
        catalog = jfli.io.Catalog.from_parquet(path)
        field_name = catalog.field[0].name if catalog.field[0].name else Path(path).stem
        field_type = type(catalog.field[0]).__name__
        st.session_state[CATALOGS_KEY].append(
            {
                "path": path,
                "label": field_name,
                "catalog": catalog,
                "field_type": field_type,
                "is_spectra": field_type in _SPECTRA_TYPES,
                "active": True,
            }
        )
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")


def _toggle_active(index: int, key: str):
    st.session_state[CATALOGS_KEY][index]["active"] = st.session_state[key]


def _update_label(index: int, key: str):
    st.session_state[CATALOGS_KEY][index]["label"] = st.session_state[key]


def _update_name(path: str, new_name: str):
    from datasets import load_dataset

    try:
        ds = load_dataset("parquet", data_files=path, split="train")
        ds = ds.map(
            lambda ex, i: {**ex, "name": new_name} if i == 0 else ex, with_indices=True
        )
        ds.to_parquet(path)
        st.session_state[
            "_rename_info"
        ] = f"Renamed to '{new_name}' in {Path(path).name}"
    except Exception as e:
        st.session_state["_rename_info"] = f"Rename failed: {e}"


def _remove_entry(index: int):
    st.session_state[CATALOGS_KEY].pop(index)


# ---------------------------------------------------------------------------
# Section 1: File Loading
# ---------------------------------------------------------------------------


def _render_file_loading(entries: list[dict]) -> None:
    with st.container(border=True):
        st.subheader("Load Parquet Files")
        col_path, col_btn = st.columns([5, 1])
        with col_path:
            new_path = st.text_input(
                "Parquet file path or folder",
                key="analysis_new_path",
                placeholder="/path/to/file.parquet  or  /folder/  or  /folder/**/spectra*.parquet",
                label_visibility="collapsed",
            )
        with col_btn:
            st.button(
                "Load",
                key="analysis_load_btn",
                on_click=_load_catalog,
                args=(new_path,),
            )

        for i, entry in enumerate(entries):
            cb, cl, cp, ct, un, cr = st.columns([0.5, 2, 3, 1.5, 0.5, 0.5])
            with cb:
                st.checkbox(
                    f"**#{i+1}**",
                    value=entry.get("active", True),
                    key=f"analysis_active_{i}",
                    on_change=_toggle_active,
                    args=(i, f"analysis_active_{i}"),
                )
            with cl:
                st.text_input(
                    "Label",
                    value=entry["label"],
                    key=f"analysis_label_{i}",
                    on_change=_update_label,
                    args=(i, f"analysis_label_{i}"),
                    label_visibility="collapsed",
                )
            with cp:
                st.text_input(
                    "Path",
                    value=entry["path"],
                    key=f"analysis_path_{i}",
                    disabled=True,
                    label_visibility="collapsed",
                )
            with ct:
                st.caption(entry["field_type"])
            with un:
                st.button(
                    "\u270E",
                    key=f"analysis_rename_{i}",
                    on_click=_update_name,
                    args=(entry["path"], entry["label"]),
                )
            with cr:
                st.button(
                    "\u2716", key=f"analysis_rm_{i}", on_click=_remove_entry, args=(i,)
                )

        if not entries:
            st.info("No files loaded. Enter a parquet file path and click Load.")

        if msg := st.session_state.pop("_rename_info", None):
            st.info(msg)


# ---------------------------------------------------------------------------
# Section 2: Field Map Visualization
# ---------------------------------------------------------------------------


def _render_field_map_section(entries: list[dict]) -> None:
    st.divider()
    st.subheader("Visualization")

    file_labels = [f"[{i+1}] {e['label']}" for i, e in enumerate(entries)]
    selected_idx = st.selectbox(
        "Select file to display",
        range(len(entries)),
        format_func=lambda i: file_labels[i],
        key="analysis_vis_select",
    )

    selected_entry = entries[selected_idx]
    field = selected_entry["catalog"].field[0]
    field_type_str = selected_entry["field_type"]
    _single = not field.is_batched()

    # Invalidate cached PNG when the selected file changes
    _fkey = selected_entry["path"]
    if st.session_state.get("_field_cache_path") != _fkey:
        st.session_state.pop("analysis_field_png", None)
        st.session_state["_field_cache_path"] = _fkey

    is_spectra = selected_entry.get("is_spectra", False)

    if is_spectra:
        st.info("Precomputed spectra — field rendering disabled.")
    else:
        with st.container(border=True):
            st.markdown("**Field Map Settings**")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                map_index = st.text_input(
                    "Index (numpy-like)",
                    value=":",
                    key="analysis_map_index",
                    disabled=_single,
                    help="Greyed out when field has a single element.",
                )
                map_ncols = st.number_input(
                    "Columns",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key="analysis_map_ncols",
                )
                _proj_disabled = field_type_str in (
                    _DENSITY_3D | _PARTICLE_TYPE | _FLAT_TYPES
                )
                map_projection = st.selectbox(
                    "Projection",
                    ["mollweide", "cart", "polar", "aitoff", "hammer", "lambert"],
                    key="analysis_map_proj",
                    disabled=_proj_disabled,
                )
            with mc2:
                map_cmap = st.selectbox(
                    "Colormap",
                    [
                        "magma",
                        "viridis",
                        "inferno",
                        "plasma",
                        "cividis",
                        "coolwarm",
                        "RdBu_r",
                        "hot",
                        "bone",
                        "gray",
                    ],
                    key="analysis_map_cmap",
                )
                map_border = st.number_input(
                    "Border width",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    key="analysis_map_border",
                    disabled=_proj_disabled,
                )
            with mc3:
                map_fig_w = st.number_input(
                    "Width/map",
                    min_value=2.0,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key="analysis_map_fig_w",
                )
                map_fig_h = st.number_input(
                    "Height/map",
                    min_value=2.0,
                    max_value=12.0,
                    value=4.0,
                    step=0.5,
                    key="analysis_map_fig_h",
                )
            with mc4:
                map_colorbar = st.checkbox(
                    "Colorbar", value=True, key="analysis_map_cbar"
                )
                map_ticks = st.checkbox(
                    "Graticule",
                    value=False,
                    key="analysis_map_ticks",
                    disabled=_proj_disabled,
                )
            with mc5:
                map_use_vmin = st.checkbox(
                    "Custom vmin", value=False, key="analysis_map_use_vmin"
                )
                map_vmin = st.number_input(
                    "vmin",
                    value=0.0,
                    format="%.4f",
                    key="analysis_map_vmin",
                    disabled=not map_use_vmin,
                )
                map_use_vmax = st.checkbox(
                    "Custom vmax", value=False, key="analysis_map_use_vmax"
                )
                map_vmax = st.number_input(
                    "vmax",
                    value=1.0,
                    format="%.4f",
                    key="analysis_map_vmax",
                    disabled=not map_use_vmax,
                )
                apply_fn = st.text_input(
                    "Apply function",
                    value="",
                    key="analysis_map_apply_fn",
                    help="Optional numpy expression, e.g. 'np.log10(x + 1e-5)'. Use 'x' as variable.",
                )

            tc1, tc2 = st.columns([3, 1])
            with tc1:
                map_title_template = st.text_input(
                    "Panel title template",
                    value="%l% - %i%",
                    key="analysis_map_title_template",
                    help="%l% = Label | %i% = Index | %r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
                )
            with tc2:
                map_dpi = st.number_input(
                    "Render DPI",
                    min_value=50,
                    max_value=2000,
                    value=100,
                    step=25,
                    key="analysis_map_dpi",
                )

            # Type-specific expandable options
            d_params = {}
            p_params = {}
            if field_type_str in _DENSITY_3D:
                with st.expander("3D Plot Options"):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        d_params["elev"] = st.number_input(
                            "Elevation", value=40.0, step=5.0, key="analysis_d_elev"
                        )
                        d_params["levels"] = st.number_input(
                            "Levels",
                            min_value=4,
                            max_value=256,
                            value=64,
                            key="analysis_d_levels",
                        )
                    with dc2:
                        d_params["azim"] = st.number_input(
                            "Azimuth", value=-30.0, step=5.0, key="analysis_d_azim"
                        )
                        d_params["project_slices"] = st.number_input(
                            "Project slices",
                            min_value=1,
                            max_value=128,
                            value=10,
                            key="analysis_d_project_slices",
                        )
                    with dc3:
                        d_params["zoom"] = st.number_input(
                            "Zoom",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.8,
                            step=0.1,
                            key="analysis_d_zoom",
                        )
                        d_params["edges"] = st.checkbox(
                            "Edges", value=True, key="analysis_d_edges"
                        )

                    st.markdown("**Crop** (e.g. `:` or `10:50`)")
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        d_crop_x = st.text_input(
                            "Crop X", value=":", key="analysis_d_crop_x"
                        )
                    with cc2:
                        d_crop_y = st.text_input(
                            "Crop Y", value=":", key="analysis_d_crop_y"
                        )
                    with cc3:
                        d_crop_z = st.text_input(
                            "Crop Z", value=":", key="analysis_d_crop_z"
                        )

                    pc1, pc2 = st.columns([1, 1])
                    with pc1:
                        d_params["do_project"] = st.checkbox(
                            "Project to 2D", value=False, key="analysis_d_do_project"
                        )
                    with pc2:
                        d_params["nz_slices"] = st.number_input(
                            "nz_slices",
                            min_value=1,
                            max_value=128,
                            value=10,
                            key="analysis_d_nz_slices",
                            disabled=not d_params["do_project"],
                        )

                    for dim, raw in [("x", d_crop_x), ("y", d_crop_y), ("z", d_crop_z)]:
                        try:
                            d_params[f"crop_{dim}"] = eval(f"np.s_[{raw}]", {"np": np})
                        except Exception:
                            d_params[f"crop_{dim}"] = slice(None)
                    d_params["crop"] = (
                        d_params.pop("crop_x"),
                        d_params.pop("crop_y"),
                        d_params.pop("crop_z"),
                    )

            elif field_type_str in _PARTICLE_TYPE:
                with st.expander("Particle Plot Options"):
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        p_params["thinning"] = st.number_input(
                            "Thinning",
                            min_value=1,
                            max_value=64,
                            value=4,
                            key="analysis_p_thinning",
                        )
                        p_params["elev"] = st.number_input(
                            "Elevation", value=40.0, step=5.0, key="analysis_p_elev"
                        )
                    with pc2:
                        p_params["point_size"] = st.number_input(
                            "Point size",
                            min_value=0.5,
                            max_value=50.0,
                            value=5.0,
                            step=0.5,
                            key="analysis_p_point_size",
                        )
                        p_params["azim"] = st.number_input(
                            "Azimuth", value=-30.0, step=5.0, key="analysis_p_azim"
                        )
                    with pc3:
                        p_params["alpha"] = st.slider(
                            "Alpha",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            step=0.05,
                            key="analysis_p_alpha",
                        )
                        p_params["zoom"] = st.number_input(
                            "Zoom",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.8,
                            step=0.1,
                            key="analysis_p_zoom",
                        )
                    pw1, pw2 = st.columns(2)
                    with pw1:
                        weights_raw = st.text_input(
                            "Weights",
                            value="",
                            key="analysis_p_weights",
                            help="Leave empty for none, or enter: 'redshift', 'z', 'scale', 'a', 'comoving', 'r', or a float.",
                        )
                    with pw2:
                        p_params["weights_title"] = (
                            st.text_input(
                                "Weights title",
                                value="",
                                key="analysis_p_weights_title",
                                help="Optional colorbar label.",
                            )
                            or None
                        )
                    if not weights_raw:
                        p_params["weights"] = None
                    else:
                        try:
                            p_params["weights"] = float(weights_raw)
                        except ValueError:
                            p_params["weights"] = weights_raw

            plot_btn = st.button("Plot", key="analysis_plot_btn", type="primary")

        with st.container(border=True):
            st.markdown("**Field Map**")

            if plot_btn:
                # Index slicing
                if _single:
                    plot_field = field
                else:
                    try:
                        idx = eval(f"np.s_[{map_index}]", {"np": np})
                        plot_field = field[idx]
                    except Exception as e:
                        st.error(f"Invalid index '{map_index}': {e}")
                        plot_field = field

                # Apply function
                if apply_fn.strip():
                    try:
                        parsed_fn = eval(f"lambda x: {apply_fn}", {"np": np})
                        plot_field = plot_field.apply_fn(parsed_fn)
                    except Exception as e:
                        st.error(f"Apply function error: {e}")

                map_params = {
                    "ncols": int(map_ncols),
                    "cmap": map_cmap,
                    "fig_w": float(map_fig_w),
                    "fig_h": float(map_fig_h),
                    "colorbar": map_colorbar,
                    "ticks": map_ticks,
                    "vmin": map_vmin if map_use_vmin else None,
                    "vmax": map_vmax if map_use_vmax else None,
                    "border": float(map_border),
                    "projection": map_projection,
                    "dpi": int(map_dpi),
                    "title_template": map_title_template,
                    "label": selected_entry["label"],
                }

                png = None
                fig = None
                if field_type_str in _DENSITY_3D:
                    from . import density_analysis_compute

                    png, fig = density_analysis_compute.render_density_field_map(
                        selected_entry, plot_field, map_params, d_params
                    )
                elif field_type_str in _PARTICLE_TYPE:
                    from . import density_analysis_compute

                    png, fig = density_analysis_compute.render_particle_field_map(
                        selected_entry, plot_field, map_params, p_params
                    )
                elif field_type_str in _SPHERICAL_TYPES:
                    from . import spherical_analysis_form

                    png, fig = spherical_analysis_form.render_field_map(
                        selected_entry, plot_field, map_params
                    )
                else:  # flat types
                    from . import flat_analysis

                    png, fig = flat_analysis.render_flat_field_map(
                        selected_entry, plot_field, map_params
                    )

                if png is not None:
                    st.session_state["analysis_field_png"] = png
                    st.session_state["analysis_field_fig"] = fig
                else:
                    st.error(
                        "Field map rendering failed — check the console for details."
                    )

            field_png = st.session_state.get("analysis_field_png")
            field_fig = st.session_state.get("analysis_field_fig")
            if field_png:
                st.image(field_png)
                if field_fig is not None:
                    from app.components.save_figure import render_save_figure

                    render_save_figure(
                        field_fig, key_prefix="field_map", filename="field_map"
                    )
            else:
                st.info("Adjust settings above, then click **Plot**.")

    # Metadata mini-plots (fast, no heavy compute)
    meta_attrs = [
        (a, lbl, u)
        for a, lbl, u in [
            ("comoving_centers", "Comoving Centers", "Mpc/h"),
            ("scale_factors", "Scale Factors", "a"),
            ("z_sources", "z Sources", "z"),
            ("density_width", "Density Width", "Mpc/h"),
        ]
        if getattr(field, a, None) is not None
    ]
    if meta_attrs:
        # 1. Logic to render the plots into a single PNG buffer
        meta_png = None
        with _plt_lock:
            # Use a slightly wider figsize for the 2x2 grid
            fig_m, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes_flat = axes.flatten()

            for i, (attr, lbl, unit) in enumerate(meta_attrs):
                ax = axes_flat[i]
                arr = np.asarray(getattr(field, attr))

                if arr.ndim == 0:
                    ax.axhline(float(arr), color="C0", linestyle="--")
                    ax.text(
                        0.5,
                        float(arr),
                        f"{float(arr):.2f}",
                        transform=ax.get_yaxis_transform(),
                        ha="center",
                        va="bottom",
                    )
                else:
                    ax.plot(arr, marker="o", markersize=4, linewidth=1.5)
                    ax.set_xticks(np.arange(len(arr)))
                    ax.grid(True, linestyle="--", alpha=0.6)

                ax.set_title(lbl, fontsize=11, fontweight="bold")
                ax.set_xlabel("Shell" if arr.ndim > 0 else "", fontsize=9)
                ax.set_ylabel(unit, fontsize=9)
                ax.tick_params(labelsize=8)

            # Hide unused quadrants
            for j in range(len(meta_attrs), 4):
                axes_flat[j].axis("off")

            fig_m.tight_layout()

            # Use your utility function from utils.py
            meta_png = _fig_to_png(fig_m, dpi=600)
            plt.close(fig_m)

        # 2. Display the rendered PNG in a clean container
        if meta_png:
            st.caption("Field Metadata Analysis")
            st.image(meta_png)

    with st.expander("Field info"):
        st.code(repr(field), language=None)


# ---------------------------------------------------------------------------
# Section 3: Power Spectra routing
# ---------------------------------------------------------------------------


def _render_spectra_section(active_entries: list[dict]) -> None:
    st.divider()
    st.subheader("Power Spectra")

    ref_entry = active_entries[0]
    ref_field_type = ref_entry["field_type"]

    # Precomputed spectra: route to the shared tab functions with precomputed=True
    if ref_entry.get("is_spectra", False):
        from jax_fli._src.base._enums import SpectralUnit

        cl_entries = [
            e
            for e in active_entries
            if getattr(e["catalog"].field[0], "unit", None)
            != SpectralUnit.POWER_SPECTRA
        ]
        pk_entries = [
            e
            for e in active_entries
            if getattr(e["catalog"].field[0], "unit", None)
            == SpectralUnit.POWER_SPECTRA
        ]
        tab_cl, tab_pk = st.tabs(["Angular Cl", "3D P(k)"])
        with tab_cl:
            if cl_entries:
                from . import spherical_analysis_form

                spherical_analysis_form.cl_tab(
                    cl_entries, cl_entries[0]["field_type"], precomputed=True
                )
            else:
                st.info("No Angular Cl spectra in the current selection.")
        with tab_pk:
            if pk_entries:
                from . import density_analysis_form

                density_analysis_form.pk_tab(pk_entries, precomputed=True)
            else:
                st.info("No 3D P(k) spectra in the current selection.")
        return

    tab_cl, tab_pk = st.tabs(["Angular Cl", "3D P(k)"])

    with tab_cl:
        if ref_field_type in (_DENSITY_3D | _PARTICLE_TYPE):
            msg = (
                f"Angular Cl not supported for **{ref_field_type}**. "
                "Use the **3D P(k)** tab for DensityField."
                if ref_field_type in _DENSITY_3D
                else f"Angular Cl not supported for **{ref_field_type}**."
            )
            st.error(msg)
        elif ref_field_type in _FLAT_TYPES:
            st.error(
                f"Angular Cl is not supported for **{ref_field_type}** (flat types). "
                "Use a spherical field type for Cl analysis."
            )
        else:
            from . import spherical_analysis_form

            spherical_analysis_form.cl_tab(active_entries, ref_field_type)

    with tab_pk:
        if ref_field_type not in _DENSITY_3D:
            st.error(
                f"3D P(k) only supported for **DensityField**. "
                f"Current field type: **{ref_field_type}**. "
                "Use the **Angular Cl** tab for spherical/flat fields."
            )
        else:
            from . import density_analysis_form

            density_analysis_form.pk_tab(active_entries)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Render the full Analysis page. Called by the Streamlit entry point."""
    _init_session_state()
    entries = st.session_state[CATALOGS_KEY]

    _render_file_loading(entries)

    if not entries:
        st.stop()

    _render_field_map_section(entries)

    active_entries = [e for e in entries if e.get("active", True)]
    if not active_entries:
        st.warning("No active entries selected.")
        st.stop()

    _render_spectra_section(active_entries)
