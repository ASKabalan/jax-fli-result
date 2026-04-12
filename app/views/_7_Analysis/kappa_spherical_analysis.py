"""SphericalKappaField angular Cl analysis: computation, theory, and Cl tab UI.

The staircase figure builders are identical to those in spherical_analysis and
are re-exported here — there is a single source of truth for that layout code.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from .utils import (
    _fig_to_png,
    _plt_lock,
    parse_shell_index,
    pixel_window_function,
)
# Re-export all figure builders — kappa uses the same staircase layout.
from .spherical_analysis import (
    _CL_BUILDERS,
    _build_cl_main_only,
    _build_cl_with_ref_ratio,
    _build_cl_with_theory_ratio,
    _build_cl_with_both_ratios,
    _build_cl_ratio_only_ref,
    _build_cl_ratio_only_theory,
    _build_cl_ratio_only_both,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def compute_cls(
    active_entries: list[dict],
    lmin: int,
    lmax: int,
) -> list[tuple[str, object]]:
    """Compute angular Cl for SphericalKappaField entries."""
    spectra_results = []
    for entry in active_entries:
        fld = entry["catalog"].field[0]
        cl  = fld.angular_cl(lmax=int(lmax), method="healpy")[..., int(lmin):]
        spectra_results.append((entry["label"], cl))
    return spectra_results


def compute_theory_cl(
    ref_obj,
    ells,
    nl_fn_name: str,
    nz_shear_mode: str,
    z_sources: list[float],
    apply_pixwin: bool,
    lmin: int,
    lmax: int,
) -> object | None:
    """Compute theory Cl for SphericalKappaField (weak lensing probe).

    Parameters
    ----------
    nz_shear_mode:
        ``"s3"`` to use the Stage-III n(z) shear, or ``"point sources"`` to
        use the explicit ``z_sources`` list.
    z_sources:
        Redshifts for point-source mode.
    apply_pixwin:
        Divide theory by the healpy pixel window² if True.
    """
    import jax_cosmo as jc
    import jax_fli as jfli

    ref_field_obj = ref_obj.field[0]
    ref_cosmo     = ref_obj.cosmology[0]
    nl_fn = jc.power.halofit if nl_fn_name == "halofit" else "linear"

    if nz_shear_mode == "s3":
        z_src = jfli.io.get_stage3_nz_shear()
    else:
        z_src = list(z_sources)

    theory_result = jfli.compute_theory_cl(
        ref_cosmo, ells, z_src,
        probe_type="weak_lensing", nonlinear_fn=nl_fn,
    )

    if apply_pixwin and theory_result is not None:
        ft = type(ref_field_obj).__name__
        if ft in ("SphericalKappaField",):
            import healpy as hp
            nside_val = int(ref_field_obj.nside)
            pw_full   = np.asarray(hp.pixwin(nside_val, lmax=int(lmax)))
            pw        = pw_full[int(lmin):]
            new_arr   = theory_result.array / pw**2
            theory_result = theory_result.replace(array=new_arr)
        else:
            # Flat kappa: sinc² pixel window
            ny, nx = ref_field_obj.flatsky_npix
            fs = ref_field_obj.field_size
            if hasattr(fs, "__len__"):
                size_y, size_x = float(fs[0]), float(fs[1])
            else:
                size_y = size_x = float(fs)
            pixel_scale = 0.5 * (size_y * 60.0 / ny + size_x * 60.0 / nx)
            pw      = pixel_window_function(np.asarray(theory_result.wavenumber), pixel_scale)
            new_arr = theory_result.array / pw**2
            theory_result = theory_result.replace(array=new_arr)

    return theory_result


# ---------------------------------------------------------------------------
# Helpers for point-source list (used as on_change callbacks in cl_tab)
# ---------------------------------------------------------------------------

def _add_nz_source():
    st.session_state["analysis_nz_point_sources"].append(0.01)


def _remove_nz_source(i: int):
    st.session_state["analysis_nz_point_sources"].pop(i)


def _update_nz_source(i: int, key: str):
    st.session_state["analysis_nz_point_sources"][i] = st.session_state[key]


# ---------------------------------------------------------------------------
# Cl tab UI — kappa variant (adds nz_shear and point-source widgets)
# ---------------------------------------------------------------------------

def cl_tab(active_entries: list[dict], ref_field_type: str) -> None:
    """Render the full Angular Cl tab for SphericalKappaField."""
    import jax.numpy as jnp

    all_types = {e["field_type"] for e in active_entries}
    if len(all_types) > 1:
        st.error("Spectra plotting only supported when all fields are the same type. "
                 f"Found: {', '.join(sorted(all_types))}")
        return

    spec_params_col, spec_plot_col = st.columns([1, 3])

    with spec_params_col:
        with st.container(border=True):
            st.markdown("**Parameters**")

            default_nside = getattr(active_entries[0]["catalog"].field[0], "nside", 512) or 512
            lc1, lc2 = st.columns(2)
            with lc1:
                lmin = st.number_input("LMIN", min_value=0, value=10, key="analysis_lmin")
            with lc2:
                lmax = st.number_input("LMAX", min_value=10, value=int(3 * default_nside),
                                       key="analysis_lmax")

            nonlinear_fn_name = st.selectbox("Nonlinear fn", ["halofit", "linear"],
                                              key="analysis_nl_fn")

            compare_ref = st.checkbox("Compare against reference", value=False,
                                      key="analysis_compare_ref",
                                      disabled=(len(active_entries) < 2))
            compare_theory = st.checkbox("Compare against theory", value=False,
                                          key="analysis_compare_theory")
            ratio_only = st.checkbox("Ratio only (hide main panel)", value=False,
                                     key="analysis_ratio_only_cl",
                                     disabled=not (compare_ref or compare_theory),
                                     help="Show only ratio panels without the main Cl panel.")

            # Kappa-specific: source redshift distribution
            nz_shear_mode = None
            if compare_theory:
                st.markdown("**Kappa settings**")
                nz_shear_mode = st.radio("nz_shear", ["s3", "point sources"],
                                         key="analysis_nz_shear_mode")
                if nz_shear_mode == "point sources":
                    st.markdown("*Source redshifts:*")
                    z_sources_list = st.session_state["analysis_nz_point_sources"]
                    for _zi, _zv in enumerate(z_sources_list):
                        _zcol, _zrm = st.columns([4, 1])
                        with _zcol:
                            st.number_input(
                                f"z_{_zi}", value=float(_zv), min_value=0.001,
                                step=0.01, format="%.3f",
                                key=f"analysis_nz_src_{_zi}",
                                label_visibility="collapsed",
                                on_change=_update_nz_source,
                                args=(_zi, f"analysis_nz_src_{_zi}"),
                            )
                        with _zrm:
                            st.button("\u2716", key=f"analysis_nz_rm_{_zi}",
                                      on_click=_remove_nz_source, args=(_zi,))
                    st.button("+ Add source", key="analysis_nz_add",
                              on_click=_add_nz_source)

            # Pixel window correction
            apply_pixwin = False
            if compare_theory:
                apply_pixwin = st.checkbox(
                    "Divide theory by pixel window", value=False,
                    key="analysis_pixwin",
                    help="Uses healpy.pixwin(nside) for spherical kappa.",
                )

            st.markdown("**Shell selection**")
            cached = st.session_state.get("analysis_spectra_results")
            if cached:
                _ns = cached[0][1].array.shape[0] if cached[0][1].array.ndim > 1 else 1
            else:
                _ns = None
            _single_shell = (_ns == 1)
            cl_index = st.text_input(
                "Shell index (numpy-style)", value=":",
                key="analysis_cl_index",
                disabled=_single_shell,
                help="Examples: ':' (all), '0:6', '::2', '-3:'. "
                     "Disabled when there is only one shell.",
            )

            st.markdown("**Plot layout**")
            spec_fig_w   = st.number_input("Width/col", min_value=2.0, max_value=16.0,
                                           value=5.0, step=0.5, key="analysis_spec_fig_w")
            spec_main_h  = st.number_input("Main panel height", min_value=1.0, max_value=10.0,
                                           value=3.0, step=0.5, key="analysis_spec_main_h")
            spec_ratio_h = st.number_input("Ratio panel height", min_value=0.5, max_value=5.0,
                                           value=1.0, step=0.25, key="analysis_spec_ratio_h")
            spec_dpi     = st.number_input("Render DPI", min_value=50, max_value=300,
                                           value=100, step=25, key="analysis_spec_dpi")
            title_template = st.text_input(
                "Panel title template", value="χ %r% Mpc/h",
                key="analysis_title_template",
                help="%r% = comoving distance  |  %z% = redshift  |  %a% = scale factor",
            )

            st.markdown("**Shading bands** (set to 0 to disable)")
            bc1, bc2 = st.columns(2)
            with bc1:
                band_2 = st.number_input("±2% band", min_value=0.0, max_value=100.0,
                                         value=2.0, step=1.0, key="analysis_band_2")
            with bc2:
                band_5 = st.number_input("±5% band", min_value=0.0, max_value=100.0,
                                         value=5.0, step=1.0, key="analysis_band_5")
            bands = [v / 100 for v in [band_2, band_5] if v > 0]

            has_spectra = bool(st.session_state.get("analysis_spectra_results"))
            cb1, cb2 = st.columns(2)
            with cb1:
                compute_btn = st.button("Compute", key="analysis_compute_btn", type="primary")
            with cb2:
                redraw_btn  = st.button("Redraw", key="analysis_redraw_btn",
                                        disabled=not has_spectra,
                                        help="Re-render from cached Cl values without recomputing")

    with spec_plot_col:
        if compute_btn:
            spectra_results = []
            theory_result   = None

            with st.spinner("Computing angular power spectra..."):
                spectra_results = compute_cls(active_entries, int(lmin), int(lmax))

            if compare_theory and spectra_results:
                ells = jnp.asarray(spectra_results[0][1].wavenumber)
                z_sources = list(st.session_state.get("analysis_nz_point_sources", [0.5]))
                with st.spinner("Computing theory Cl..."):
                    theory_result = compute_theory_cl(
                        active_entries[0]["catalog"],
                        ells, nonlinear_fn_name,
                        nz_shear_mode or "s3", z_sources,
                        apply_pixwin, int(lmin), int(lmax),
                    )

            st.session_state["analysis_spectra_results"] = spectra_results
            st.session_state["analysis_theory_result"]   = theory_result

        spectra_results = st.session_state.get("analysis_spectra_results")
        theory_result   = st.session_state.get("analysis_theory_result")

        if (compute_btn or redraw_btn) and spectra_results:
            ns = spectra_results[0][1].array.shape[0] if spectra_results[0][1].array.ndim > 1 else 1
            selected_shells = parse_shell_index(cl_index if ns > 1 else ":", ns)
            if not selected_shells:
                st.warning("No shells selected by the given index.")
            else:
                layout_params = {
                    "spec_fig_w":   spec_fig_w,
                    "spec_main_h":  spec_main_h,
                    "spec_ratio_h": spec_ratio_h,
                }
                eff_compare_ref    = compare_ref and len(spectra_results) > 1
                eff_compare_theory = compare_theory and theory_result is not None
                eff_ratio_only     = ratio_only and (eff_compare_ref or eff_compare_theory)
                key     = (eff_compare_ref, eff_compare_theory, eff_ratio_only)
                builder = _CL_BUILDERS.get(key, _build_cl_main_only)

                field_ref = active_entries[0]["catalog"].field[0]
                with st.spinner("Rendering..."):
                    with _plt_lock:
                        fig_cl = builder(
                            spectra_results, theory_result,
                            selected_shells, layout_params,
                            title_template, field_ref, bands,
                        )
                        old_fig = st.session_state.pop("analysis_spectra_fig", None)
                        if old_fig is not None:
                            plt.close(old_fig)
                        st.session_state["analysis_spectra_png"] = _fig_to_png(
                            fig_cl, dpi=int(spec_dpi))
                        st.session_state["analysis_spectra_fig"] = fig_cl

        spectra_png = st.session_state.get("analysis_spectra_png")
        spectra_fig = st.session_state.get("analysis_spectra_fig")
        if spectra_png:
            st.image(spectra_png)
            if spectra_fig is not None:
                from app.components.save_figure import render_save_figure
                render_save_figure(spectra_fig, key_prefix="spectra", filename="spectra")
        else:
            st.info("Click **Compute** to generate angular power spectra.")


