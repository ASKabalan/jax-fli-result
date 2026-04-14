"""Unified Angular Cl tab UI for all spherical and flat field types.

Replaces the separate ``cl_tab()`` functions in ``spherical_analysis.py``
and ``kappa_spherical_analysis.py``.

When theory comparison is enabled, a "Probe settings" expander appears:
  - density  — matter Cl via Limber (hidden for kappa fields)
  - s3       — weak-lensing with Stage-III n(z)
  - point sources — weak-lensing with user-specified source redshifts
"""
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from . import spherical_analysis_compute as _compute
from .spherical_analysis_compute import _CL_BUILDERS, _build_cl_main_only
from .utils import _KAPPA_TYPES, _fig_to_png, _make_title, _plt_lock, parse_slice

# ---------------------------------------------------------------------------
# Session-state callbacks for point-source z list
# ---------------------------------------------------------------------------


def _add_nz_source() -> None:
    st.session_state["analysis_nz_point_sources"].append(0.01)


def _remove_nz_source(i: int) -> None:
    st.session_state["analysis_nz_point_sources"].pop(i)


def _update_nz_source(i: int, key: str) -> None:
    st.session_state["analysis_nz_point_sources"][i] = st.session_state[key]


# ---------------------------------------------------------------------------
# Field map renderer
# ---------------------------------------------------------------------------


def render_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
) -> tuple[bytes, object] | tuple[None, None]:
    """Render a spherical (SphericalDensity / SphericalKappaField) field map; return (png_bytes, fig).

    Parameters
    ----------
    selected_entry:
        Catalog entry dict with at least ``"label"`` key.
    plot_field:
        The (possibly sliced) field object with a ``.plot()`` method.
    map_params:
        Dict of rendering settings::

            ncols, cmap, fig_w, fig_h, colorbar, ticks,
            vmin, vmax, border, projection, dpi,
            title_template, label

    Returns
    -------
    (png_bytes, fig) on success; (None, None) on failure.
    The caller is responsible for closing the figure after use.
    """
    data_arr = np.asarray(plot_field.array)
    n_maps = 1 if data_arr.ndim <= 1 else int(np.prod(data_arr.shape[:-1]))
    ncols = int(map_params["ncols"])
    nrows = max(1, ceil(n_maps / ncols))

    titles = []
    for i in range(n_maps):
        t = (
            map_params["title_template"]
            .replace("%l%", selected_entry["label"])
            .replace("%i%", str(i))
        )
        t = _make_title(t, plot_field, i)
        titles.append(t)

    fig = None
    with _plt_lock:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(map_params["fig_w"] * ncols, map_params["fig_h"] * nrows),
        )
        try:
            plot_field.plot(
                ax=axes,
                titles=titles,
                border_linewidth=map_params["border"],
                cmap=map_params["cmap"],
                colorbar=map_params["colorbar"],
                show_ticks=map_params["ticks"],
                projection_type=map_params["projection"],
                vmin=map_params["vmin"],
                vmax=map_params["vmax"],
            )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
            return png, fig
        except Exception as e:
            print(f"[spherical_analysis_form] Field map rendering failed: {e}")
            if fig is not None:
                plt.close(fig)
            return None, None


# ---------------------------------------------------------------------------
# Unified Cl tab
# ---------------------------------------------------------------------------


def cl_tab(
    active_entries: list[dict],
    ref_field_type: str,
    precomputed: bool = False,
) -> None:
    """Render the Angular Cl tab for any spherical or flat field type.

    When ``precomputed=True`` the spectra are already stored in the catalog
    (``PowerSpectrum`` objects) and are only sliced to the chosen ell range —
    no heavy computation is triggered.
    """
    import jax.numpy as jnp

    all_types = {e["field_type"] for e in active_entries}
    if not precomputed and len(all_types) > 1:
        st.error(
            "Spectra plotting only supported when all fields are the same type. "
            f"Found: {', '.join(sorted(all_types))}"
        )
        return

    # For precomputed catalogs, extract the stored ell range up front so we
    # can use it as lmin/lmax defaults and validate user input against it.
    _lmin_stored = _lmax_stored = _ells_full = None
    if precomputed:
        _ref_ps = active_entries[0]["catalog"].field[0]
        _ells_full = np.asarray(_ref_ps.wavenumber)
        _lmin_stored = int(_ells_full[0]) if _ells_full.size > 0 else 0
        _lmax_stored = int(_ells_full[-1]) if _ells_full.size > 0 else 500

    spec_params_col, spec_plot_col = st.columns([1, 3])

    with spec_params_col:
        with st.container(border=True):
            st.markdown("**Parameters**")

            # --- Reference entry selector ---
            _labels = [e["label"] for e in active_entries]
            _ref_label = st.selectbox(
                "Reference entry",
                _labels,
                index=0,
                key="analysis_spherical_cl_ref_entry",
            )
            _ref_idx = _labels.index(_ref_label)
            active_entries = [active_entries[_ref_idx]] + [
                e for i, e in enumerate(active_entries) if i != _ref_idx
            ]
            ref_field_type = active_entries[0]["field_type"]
            is_kappa = ref_field_type in _KAPPA_TYPES

            # --- ℓ range ---
            _default_nside = (
                getattr(active_entries[0]["catalog"].field[0], "nside", 512) or 512
            )
            lc1, lc2 = st.columns(2)
            with lc1:
                lmin = st.number_input(
                    "LMIN",
                    min_value=0,
                    value=(_lmin_stored or 0) if precomputed else 10,
                    key="analysis_lmin",
                )
            with lc2:
                lmax = st.number_input(
                    "LMAX",
                    min_value=10,
                    value=(_lmax_stored or int(3 * _default_nside))
                    if precomputed
                    else int(3 * _default_nside),
                    key="analysis_lmax",
                )

            nonlinear_fn_name = st.selectbox(
                "Nonlinear fn",
                ["halofit", "linear"],
                key="analysis_nl_fn",
            )

            # --- Comparison toggles ---
            compare_ref = st.checkbox(
                "Compare against reference",
                value=False,
                key="analysis_compare_ref",
                disabled=(len(active_entries) < 2),
            )
            compare_theory = st.checkbox(
                "Compare against theory",
                value=False,
                key="analysis_compare_theory",
            )
            ratio_only = st.checkbox(
                "Ratio only (hide main panel)",
                value=False,
                key="analysis_ratio_only_cl",
                disabled=not (compare_ref or compare_theory),
                help="Show only ratio panels without the main Cl panel.",
            )

            # --- Probe settings (expander, visible when compare_theory) ---
            # TODO add prove DESY3
            probe_type = "s3" if is_kappa else "density"
            nz_zmax = 2.0
            apply_pixwin = False

            if compare_theory:
                with st.expander("Probe settings", expanded=True):
                    # Density is not meaningful for kappa fields
                    _probe_options = (
                        ["s3", "point sources"]
                        if is_kappa
                        else ["density", "s3", "point sources"]  # TODO add DESY3 here
                    )
                    _pt_key = "analysis_probe_type"
                    # Reset if stored value is no longer a valid option
                    if st.session_state.get(_pt_key) not in _probe_options:
                        st.session_state[_pt_key] = _probe_options[0]
                    probe_type = st.radio(
                        "Probe",
                        _probe_options,
                        key=_pt_key,
                        help=(
                            "density: matter Cl via Limber  |  "
                            "s3: weak-lensing Stage-III n(z)  |  "
                            "point sources: explicit source redshifts"
                        ),
                    )

                    if probe_type == "density":
                        nz_zmax = st.number_input(
                            "nz_zmax",
                            min_value=0.1,
                            value=2.0,
                            step=0.1,
                            format="%.1f",
                            key="analysis_nz_zmax",
                        )
                    elif probe_type == "point sources":
                        st.markdown("*Source redshifts:*")
                        _z_list = st.session_state["analysis_nz_point_sources"]
                        for _zi, _zv in enumerate(_z_list):
                            _zcol, _zrm = st.columns([4, 1])
                            with _zcol:
                                st.number_input(
                                    f"z_{_zi}",
                                    value=float(_zv),
                                    min_value=0.001,
                                    step=0.01,
                                    format="%.3f",
                                    key=f"analysis_nz_src_{_zi}",
                                    label_visibility="collapsed",
                                    on_change=_update_nz_source,
                                    args=(_zi, f"analysis_nz_src_{_zi}"),
                                )
                            with _zrm:
                                st.button(
                                    "\u2716",
                                    key=f"analysis_nz_rm_{_zi}",
                                    on_click=_remove_nz_source,
                                    args=(_zi,),
                                )
                        st.button(
                            "+ Add source",
                            key="analysis_nz_add",
                            on_click=_add_nz_source,
                        )

                    apply_pixwin = st.checkbox(
                        "Multiply theory by pixel window",
                        value=False,
                        key="analysis_pixwin",
                        help=(
                            "Spherical: healpy.pixwin(nside)  |  "
                            "Flat: sinc² pixel window from pixel scale"
                        ),
                    )

            # --- Shell selection (applied at compute time, not plot time) ---
            st.markdown("**Shell selection**")
            _ref_arr = active_entries[0]["catalog"].field[0].array
            _ns_catalog = int(_ref_arr.shape[0]) if _ref_arr.ndim > 1 else 1
            cl_index = st.text_input(
                "Shell index (numpy-style)",
                value=":",
                key="analysis_cl_index",
                disabled=(_ns_catalog == 1),
                help="Examples: ':' (all), '0:6', '::2', '-3:'. "
                "Applied when you click Compute / Plot — re-click to update.",
            )

            # --- Plot layout ---
            st.markdown("**Plot layout**")
            spec_fig_w = st.number_input(
                "Width/col",
                min_value=2.0,
                max_value=16.0,
                value=5.0,
                step=0.5,
                key="analysis_spec_fig_w",
            )
            spec_main_h = st.number_input(
                "Main height",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="analysis_spec_main_h",
            )
            spec_ratio_h = st.number_input(
                "Ratio height",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="analysis_spec_ratio_h",
            )
            spec_dpi = st.number_input(
                "Render DPI",
                min_value=50,
                max_value=300,
                value=100,
                step=25,
                key="analysis_spec_dpi",
            )
            title_template = st.text_input(
                "Panel title template",
                value="χ %r% Mpc/h",
                key="analysis_title_template",
                help="%r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
            )

            # --- Shading bands ---
            st.markdown("**Shading bands** (set to 0 to disable)")
            bc1, bc2 = st.columns(2)
            with bc1:
                band_2 = st.number_input(
                    "±2% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=2.0,
                    step=1.0,
                    key="analysis_band_2",
                )
            with bc2:
                band_5 = st.number_input(
                    "±5% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    key="analysis_band_5",
                )
            bands = [v / 100 for v in [band_2, band_5] if v > 0]

            # --- Buttons ---
            _has_spectra = bool(st.session_state.get("analysis_spectra_results"))
            cb1, cb2 = st.columns(2)
            with cb1:
                compute_btn = st.button(
                    "Plot" if precomputed else "Compute",
                    key="analysis_compute_btn",
                    type="primary",
                )
            with cb2:
                redraw_btn = st.button(
                    "Redraw",
                    key="analysis_redraw_btn",
                    disabled=not _has_spectra,
                    help="Re-render from cached Cl values without recomputing",
                )

    with spec_plot_col:
        if compute_btn:
            spectra_results = []
            theory_result = None
            st.session_state.pop("analysis_spectra_results", None)
            st.session_state.pop("analysis_theory_result", None)

            selected_shells = parse_slice(cl_index)

            if precomputed:
                assert (
                    _lmin_stored is not None
                    and _lmax_stored is not None
                    and _ells_full is not None
                )
                if int(lmin) < _lmin_stored or int(lmax) > _lmax_stored:
                    st.error(
                        f"LMIN/LMAX ({lmin}–{lmax}) outside stored range "
                        f"[{_lmin_stored}, {_lmax_stored}]."
                    )
                    st.stop()
                _lmin_idx = int(np.searchsorted(_ells_full, int(lmin)))
                _lmax_idx = int(np.searchsorted(_ells_full, int(lmax), side="right"))
                spectra_results = [
                    (
                        e["label"],
                        e["catalog"].field[0][selected_shells, _lmin_idx:_lmax_idx],
                    )
                    for e in active_entries
                ]
            else:
                with st.spinner("Computing angular power spectra..."):
                    spectra_results = _compute.compute_cls(
                        active_entries, int(lmin), int(lmax), selected_shells
                    )

            # Slice to selected shells now — builders receive pre-sliced data
            if spectra_results:
                # check that all spectra have same length
                spec_lengths = [s[1].wavenumber.size for s in spectra_results]
                if len(set(spec_lengths)) > 1:
                    st.error(
                        "Spectra have different ell lengths after slicing — cannot compare."
                    )
                    st.stop()
                comoving_centers = np.array(
                    [s[1].comoving_centers for s in spectra_results]
                )
                scale_factors = np.array([s[1].scale_factors for s in spectra_results])
                z_sources = np.array([s[1].z_sources for s in spectra_results])
                density_width = np.array([s[1].density_width for s in spectra_results])

                if not np.all(np.isclose(comoving_centers, comoving_centers[0])):
                    st.warning(
                        "Spectra have different comoving centers — cannot compare."
                    )
                if not np.all(np.isclose(scale_factors, scale_factors[0])):
                    st.warning("Spectra have different scale factors — cannot compare.")
                if not np.all(np.isclose(z_sources, z_sources[0])):
                    st.warning("Spectra have different z sources — cannot compare.")
                if not np.all(np.isclose(density_width, density_width[0])):
                    st.warning(
                        "Spectra have different density widths — cannot compare."
                    )

            if compare_theory and spectra_results:
                ells = jnp.asarray(spectra_results[0][1].wavenumber)
                z_sources = list(
                    st.session_state.get("analysis_nz_point_sources", [0.5])
                )
                with st.spinner("Computing theory Cl..."):
                    theory_result = _compute.compute_theory_cl(
                        active_entries[0]["catalog"],
                        ells,
                        nonlinear_fn_name,
                        probe_type,
                        z_sources,
                        nz_zmax,
                        apply_pixwin,
                        int(lmin),
                        int(lmax),
                    )

            st.session_state["analysis_spectra_results"] = spectra_results
            st.session_state["analysis_theory_result"] = theory_result

        spectra_results = st.session_state.get("analysis_spectra_results")
        theory_result = st.session_state.get("analysis_theory_result")

        if (compute_btn or redraw_btn) and spectra_results:
            layout_params = {
                "spec_fig_w": spec_fig_w,
                "spec_main_h": spec_main_h,
                "spec_ratio_h": spec_ratio_h,
            }
            eff_compare_ref = compare_ref and len(spectra_results) > 1
            eff_compare_theory = compare_theory and theory_result is not None
            eff_ratio_only = ratio_only and (eff_compare_ref or eff_compare_theory)
            builder = _CL_BUILDERS.get(
                (eff_compare_ref, eff_compare_theory, eff_ratio_only),
                _build_cl_main_only,
            )
            with st.spinner("Rendering..."):
                with _plt_lock:
                    fig_cl = builder(
                        spectra_results,
                        theory_result,
                        layout_params,
                        title_template,
                        bands,
                    )
                    old_fig = st.session_state.pop("analysis_spectra_fig", None)
                    if old_fig is not None:
                        plt.close(old_fig)
                    st.session_state["analysis_spectra_png"] = _fig_to_png(
                        fig_cl,
                        dpi=int(spec_dpi),
                    )
                    st.session_state["analysis_spectra_fig"] = fig_cl

        spectra_png = st.session_state.get("analysis_spectra_png")
        spectra_fig = st.session_state.get("analysis_spectra_fig")
        if spectra_png:
            st.image(spectra_png)
            # TODO save figure should download the pdf make this code mirror how it is done in /home/wassim/Projects/NBody/jax-fli-result/app/views/_7_Analysis/form.py: L390
            # after making that code work and nicely clean
            if spectra_fig is not None:
                from app.components.save_figure import render_save_figure

                render_save_figure(
                    spectra_fig, key_prefix="spectra", filename="spectra"
                )
        else:
            st.info(
                "Click **Plot** to display angular power spectra."
                if precomputed
                else "Click **Compute** to generate angular power spectra."
            )
