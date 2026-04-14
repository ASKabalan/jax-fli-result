"""DensityField and ParticleField Streamlit UI: 3D P(k) tab."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .density_analysis_compute import (
    PK_BUILDERS,
    compute_pk,
    compute_theory_pk,
)
from .utils import _DENSITY_3D, _fig_to_png, _plt_lock, parse_slice


def pk_tab(
    active_entries: list[dict],
    precomputed: bool = False,
) -> None:
    """Render the full 3D P(k) tab.

    When ``precomputed=True`` the P(k) arrays are already stored in the catalog
    and are read directly — no heavy computation is triggered.

    The caller (form.py) is responsible for routing field-type errors before
    calling this function.
    """
    spec_params_pk, spec_plot_pk = st.columns([1, 3])

    with spec_params_pk:
        with st.container(border=True):
            st.markdown("**Parameters**")

            _labels = [e["label"] for e in active_entries]
            _ref_label = st.selectbox(
                "Reference entry",
                _labels,
                index=0,
                key="analysis_pk_ref_entry",
            )
            _ref_idx = _labels.index(_ref_label)
            active_entries = [active_entries[_ref_idx]] + [
                e for i, e in enumerate(active_entries) if i != _ref_idx
            ]

            pk_nl_fn = st.selectbox(
                "Nonlinear fn", ["halofit", "linear"], key="analysis_pk_nl_fn"
            )

            compare_theory_pk = st.checkbox(
                "Compare against theory", value=False, key="analysis_pk_compare_theory"
            )
            ratio_only_pk = st.checkbox(
                "Ratio only (hide main panel)",
                value=False,
                key="analysis_ratio_only_pk",
                disabled=not compare_theory_pk,
                help="Show only the ratio panel without the main P(k) panel.",
            )

            st.markdown("**Snapshot selection**")
            cached_pk = st.session_state.get("analysis_pk_results")
            if cached_pk:
                _ref_pk_arr = np.asarray(cached_pk[0][0][1].array)
                _ns_pk = _ref_pk_arr.shape[0] if _ref_pk_arr.ndim > 1 else 1
            else:
                _ns_pk = None
            _single_snap = _ns_pk == 1
            snap_index = st.text_input(
                "Snapshot index (numpy-style)",
                value=":",
                key="analysis_snap_index",
                disabled=_single_snap,
                help="Examples: ':' (all), '0:3', '-1:'. Selects which snapshots to plot.",
            )

            pk_fig_w = st.number_input(
                "Width/snapshot",
                min_value=2.0,
                max_value=16.0,
                value=5.0,
                step=0.5,
                key="analysis_pk_fig_w",
            )
            pk_main_h = st.number_input(
                "Main panel height",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="analysis_pk_main_h",
            )
            pk_ratio_h = st.number_input(
                "Ratio panel height",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="analysis_pk_ratio_h",
            )
            pk_dpi = st.number_input(
                "Render DPI",
                min_value=50,
                max_value=300,
                value=100,
                step=25,
                key="analysis_pk_dpi",
            )

            st.markdown("**Shading bands** (set to 0 to disable)")
            pb1, pb2 = st.columns(2)
            with pb1:
                pk_band_10 = st.number_input(
                    "±10% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=1.0,
                    key="analysis_pk_band_10",
                )
            with pb2:
                pk_band_20 = st.number_input(
                    "±20% band",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key="analysis_pk_band_20",
                )
            pk_bands = [v / 100 for v in [pk_band_10, pk_band_20] if v > 0]

            has_pk = bool(st.session_state.get("analysis_pk_results"))
            pk_cb1, pk_cb2 = st.columns(2)
            with pk_cb1:
                pk_compute_btn = st.button(
                    "Plot" if precomputed else "Compute",
                    key="analysis_pk_compute_btn",
                    type="primary",
                )
            with pk_cb2:
                pk_redraw_btn = st.button(
                    "Redraw",
                    key="analysis_pk_redraw_btn",
                    disabled=not has_pk,
                    help="Re-render from cached P(k) values without recomputing",
                )

    with spec_plot_pk:
        selected_snaps = parse_slice(snap_index)
        if pk_compute_btn:
            if precomputed:
                pk_results = [
                    (e["label"], e["catalog"].field[0][selected_snaps])
                    for e in active_entries
                ]
                ref_fld_pk = active_entries[0]["catalog"].field[0][selected_snaps]
                ref_cosmo_pk = active_entries[0]["catalog"].cosmology[0]
            else:
                all_types_pk = {e["field_type"] for e in active_entries}
                if all_types_pk != _DENSITY_3D:
                    st.error(
                        "3D P(k) only supported when all active fields are DensityField. "
                        f"Found: {', '.join(sorted(all_types_pk))}"
                    )
                    pk_results = ref_fld_pk = ref_cosmo_pk = None
                else:
                    with st.spinner("Computing 3D power spectra..."):
                        pk_results, ref_fld_pk, ref_cosmo_pk = compute_pk(
                            active_entries, selected_snaps
                        )

            if pk_results:
                theory_pks = None
                if compare_theory_pk:
                    with st.spinner("Computing theory P(k)..."):
                        theory_pks = compute_theory_pk(
                            ref_fld_pk, ref_cosmo_pk, pk_results, pk_nl_fn
                        )
                st.session_state["analysis_pk_results"] = (
                    pk_results,
                    theory_pks,
                    ref_fld_pk,
                )

        pk_cached = st.session_state.get("analysis_pk_results")
        if (pk_compute_btn or pk_redraw_btn) and pk_cached:
            _pk_results, _theory_pks, _ref_fld_pk = pk_cached

            layout_params = {
                "fig_w": pk_fig_w,
                "main_h": pk_main_h,
                "ratio_h": pk_ratio_h,
            }
            eff_theory = compare_theory_pk and _theory_pks is not None
            eff_ratio_only = ratio_only_pk and eff_theory
            builder = PK_BUILDERS.get(
                (eff_theory, eff_ratio_only), PK_BUILDERS[(False, False)]
            )

            with st.spinner("Rendering..."):
                with _plt_lock:
                    fig_pk = builder(
                        _pk_results,
                        _theory_pks,
                        _ref_fld_pk,
                        layout_params,
                        pk_bands,
                    )
                    st.session_state["analysis_pk_png"] = _fig_to_png(
                        fig_pk, dpi=int(pk_dpi)
                    )
                    st.session_state["analysis_pk_fig"] = fig_pk

        pk_png = st.session_state.get("analysis_pk_png")
        pk_fig = st.session_state.get("analysis_pk_fig")
        if pk_png:
            st.image(pk_png)
            if pk_fig is not None:
                from app.components.save_figure import render_save_figure

                render_save_figure(
                    pk_fig, key_prefix="pk", filename="power_spectrum_3d"
                )
        else:
            st.info(
                "Click **Plot** to display the 3D matter power spectrum."
                if precomputed
                else "Click **Compute** to generate the 3D matter power spectrum."
            )
