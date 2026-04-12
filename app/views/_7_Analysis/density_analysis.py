"""DensityField and ParticleField analysis: 3D field maps and 3D P(k).

Figure builders follow a strict "one function per plot mode" rule — no nested
ifs for routing. Three modes: main only, with theory ratio, ratio only.
"""
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .utils import (
    _PALETTE, _COLOR_THEORY,
    _fig_to_png, _make_title,
    _clean_ratio_ax,
    _plt_lock,
    parse_shell_index,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pk_ratio_ylim(ax, bands: list[float]) -> None:
    """Set ylim for a P(k) ratio panel — wider than Cl because scatter is larger."""
    margin = max(bands) * 1.5 if bands else 0.5
    ax.set_ylim(1.0 - margin, 1.0 + margin)


def _snap_title(ref_fld, snap_idx: int) -> str:
    """Return '$a=X.XX$  ($z=Y.YY$)' title string for a given snapshot index."""
    import jax_cosmo as jc
    scale_factors = None
    if ref_fld.scale_factors is not None:
        scale_factors = np.asarray(ref_fld.scale_factors)
    if scale_factors is not None:
        a = float(scale_factors[snap_idx]) if scale_factors.ndim > 0 else float(scale_factors)
    else:
        a = 1.0
    z = float(jc.utils.a2z(a))
    return f"$a={a:.2f}$  ($z={z:.2f}$)"


def _pk_arr_for_snap(pk_sim, snap_idx: int):
    arr = np.asarray(pk_sim.array)
    return arr[snap_idx] if arr.ndim > 1 else arr


def _setup_pk_fig(n_cols: int, layout_params: dict, n_rows: int = 1):
    """Create figure with n_rows × n_cols subplots for P(k) layout."""
    return plt.subplots(
        n_rows, n_cols,
        figsize=(float(layout_params["fig_w"]) * n_cols,
                 float(layout_params["main_h"]) + float(layout_params["ratio_h"])),
        squeeze=False,
    )


# ---------------------------------------------------------------------------
# Three P(k) figure builders — one per plot mode
# ---------------------------------------------------------------------------
# Uniform signature:
#   pk_results    – list of (label, PK)
#   theory_pks    – list of arrays (one per snap) or None
#   selected_snaps – list[int] snapshot indices
#   ref_fld       – reference field object (for scale factors)
#   layout_params – {"fig_w", "main_h", "ratio_h"}
#   bands         – list[float] fractional shading bands


def _build_pk_main_only(
    pk_results, theory_pks,
    selected_snaps, ref_fld, layout_params, bands,
) -> plt.Figure:
    """P(k) top panels only for each selected snapshot — no ratio row."""
    n = len(selected_snaps)
    fig, axes = plt.subplots(
        1, n,
        figsize=(float(layout_params["fig_w"]) * n, float(layout_params["main_h"])),
        squeeze=False,
    )

    for col, snap_idx in enumerate(selected_snaps):
        ax = axes[0, col]
        k  = np.asarray(pk_results[0][1].wavenumber)

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            color  = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, snap_idx)
            ax.loglog(k, pk_arr, color=color, linewidth=2, label=lbl)

        ax.set_title(_snap_title(ref_fld, snap_idx))
        if col == 0:
            ax.set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
        ax.legend(fontsize="small")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("3D Matter Power Spectrum", fontsize=14)
    fig.tight_layout()
    return fig


def _build_pk_with_theory_ratio(
    pk_results, theory_pks,
    selected_snaps, ref_fld, layout_params, bands,
) -> plt.Figure:
    """P(k) top panels + sim/theory ratio row for each selected snapshot."""
    n = len(selected_snaps)
    fig, axes = plt.subplots(
        2, n,
        figsize=(float(layout_params["fig_w"]) * n,
                 float(layout_params["main_h"]) + float(layout_params["ratio_h"])),
        sharex="col",
        gridspec_kw={"height_ratios": [float(layout_params["main_h"]),
                                       float(layout_params["ratio_h"])],
                     "hspace": 0.05},
        squeeze=False,
    )

    for col, snap_idx in enumerate(selected_snaps):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        k      = np.asarray(pk_results[0][1].wavenumber)

        # Theory
        th = np.asarray(theory_pks[snap_idx]) if theory_pks else None
        if th is not None:
            ax_top.loglog(k, th, color=_COLOR_THEORY, linestyle="--",
                          linewidth=2, label="Theory (Halofit)")

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            color  = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, snap_idx)
            ax_top.loglog(k, pk_arr, color=color, linewidth=2, label=lbl)

        ax_top.set_title(_snap_title(ref_fld, snap_idx))
        if col == 0:
            ax_top.set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
        ax_top.legend(fontsize="small")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.tick_params(labelbottom=False)

        # Ratio
        for ci, (lbl, pk_sim) in enumerate(pk_results):
            if th is None:
                break
            color  = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, snap_idx)
            ax_bot.semilogx(k, pk_arr / th, color=color, linewidth=2)

        ylabel = "Sim / Theory" if col == 0 else ""
        _clean_ratio_ax(ax_bot, ylabel, bands)
        _pk_ratio_ylim(ax_bot, bands)
        if col != 0:
            ax_bot.tick_params(labelleft=False)
        ax_bot.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

        if col == n - 1 and bands:
            ax_bot.legend(loc="lower left", fontsize="x-small")

    fig.suptitle("3D Matter Power Spectrum vs Theory", fontsize=14)
    fig.tight_layout()
    return fig


def _build_pk_ratio_only_theory(
    pk_results, theory_pks,
    selected_snaps, ref_fld, layout_params, bands,
) -> plt.Figure:
    """Sim/theory ratio panels only — no main P(k) panel."""
    n = len(selected_snaps)
    fig, axes = plt.subplots(
        1, n,
        figsize=(float(layout_params["fig_w"]) * n, float(layout_params["ratio_h"])),
        squeeze=False,
    )

    for col, snap_idx in enumerate(selected_snaps):
        ax = axes[0, col]
        k  = np.asarray(pk_results[0][1].wavenumber)
        th = np.asarray(theory_pks[snap_idx]) if theory_pks else None

        for ci, (lbl, pk_sim) in enumerate(pk_results):
            if th is None:
                break
            color  = _PALETTE[ci % len(_PALETTE)]
            pk_arr = _pk_arr_for_snap(pk_sim, snap_idx)
            ax.semilogx(k, pk_arr / th, color=color, linewidth=2, label=lbl)

        ax.set_title(_snap_title(ref_fld, snap_idx))
        ylabel = "Sim / Theory" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        _pk_ratio_ylim(ax, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

    fig.suptitle("P(k) / Theory", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PK_BUILDERS = {
    (False, False): _build_pk_main_only,
    (True,  False): _build_pk_with_theory_ratio,
    (True,  True):  _build_pk_ratio_only_theory,
}


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def compute_pk(
    active_entries: list[dict],
) -> tuple[list[tuple[str, object]], object, object]:
    """Compute 3D P(k) for all active DensityField entries.

    Returns
    -------
    (pk_results, ref_fld, ref_cosmo)
    """
    pk_results = []
    for entry in active_entries:
        fld    = entry["catalog"].field[0]
        pk_sim = fld.power()
        pk_results.append((entry["label"], pk_sim))
    ref_fld   = active_entries[0]["catalog"].field[0]
    ref_cosmo = active_entries[0]["catalog"].cosmology[0]
    return pk_results, ref_fld, ref_cosmo


def compute_theory_pk(
    ref_fld,
    ref_cosmo,
    pk_results: list[tuple[str, object]],
    nl_fn_name: str,
) -> list | None:
    """Compute theory P(k) for each snapshot present in the reference field.

    Returns a list of arrays, one per snapshot, aligned with the wavenumber
    grid of ``pk_results[0][1].wavenumber``.
    """
    import jax
    import jax_cosmo as jc

    ref_pk     = pk_results[0][1]
    ref_pk_arr = np.asarray(ref_pk.array)
    n_snaps    = ref_pk_arr.shape[0] if ref_pk_arr.ndim > 1 else 1

    sfa = None
    if ref_fld.scale_factors is not None:
        sfa = np.asarray(ref_fld.scale_factors)

    nl_fn = jc.power.halofit if nl_fn_name == "halofit" else jc.power.linear

    theory_pks = []
    for i in range(n_snaps):
        a_i = (float(sfa[i]) if (sfa is not None and sfa.ndim > 0)
               else (float(sfa) if sfa is not None else 1.0))
        th = jax.jit(
            jc.power.nonlinear_matter_power,
            static_argnames=["nonlinear_fn"],
        )(ref_cosmo, ref_pk.wavenumber, a=a_i, nonlinear_fn=nl_fn)
        theory_pks.append(th)
    return theory_pks


# ---------------------------------------------------------------------------
# Field map renderers
# ---------------------------------------------------------------------------

def render_density_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
    d_params: dict,
) -> bytes | None:
    """Render a 3D DensityField map to PNG bytes.

    Parameters
    ----------
    d_params:
        Dict with keys: ``elev``, ``azim``, ``zoom``, ``edges``, ``levels``,
        ``project_slices``, ``crop`` (3-tuple of slices), ``do_project``,
        ``nz_slices``.
    """
    data_arr = np.asarray(plot_field.array)
    n_plots  = data_arr.shape[0] if data_arr.ndim == 4 else 1
    ncols    = int(map_params["ncols"])
    nrows    = max(1, ceil(n_plots / ncols))

    titles = []
    for i in range(n_plots):
        t = (map_params["title_template"]
             .replace("%l%", selected_entry["label"])
             .replace("%i%", str(i)))
        t = _make_title(t, plot_field, i)
        titles.append(t)

    with _plt_lock:
        try:
            if d_params["do_project"]:
                projected = plot_field.project(nz_slices=int(d_params["nz_slices"]))
                fig, _ = projected.plot(
                    ncols=ncols,
                    figsize=(float(map_params["fig_w"]) * ncols,
                             float(map_params["fig_h"]) * nrows),
                    cmap=map_params["cmap"],
                    colorbar=map_params["colorbar"],
                    vmin=map_params["vmin"],
                    vmax=map_params["vmax"],
                    titles=titles,
                )
            else:
                fig, _ = plot_field.plot(
                    ncols=ncols,
                    figsize=(float(map_params["fig_w"]) * ncols,
                             float(map_params["fig_h"]) * nrows),
                    cmap=map_params["cmap"],
                    colorbar=map_params["colorbar"],
                    vmin=map_params["vmin"],
                    vmax=map_params["vmax"],
                    titles=titles,
                    elev=float(d_params["elev"]),
                    azim=float(d_params["azim"]),
                    zoom=float(d_params["zoom"]),
                    edges=bool(d_params["edges"]),
                    levels=int(d_params["levels"]),
                    project_slices=int(d_params["project_slices"]),
                    crop=d_params["crop"],
                )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
        except Exception as e:
            print(f"[density_analysis] DensityField map rendering failed: {e}")
            png = None
        finally:
            plt.close(fig)

    return png


def render_particle_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
    p_params: dict,
) -> bytes | None:
    """Render a ParticleField scatter plot to PNG bytes.

    Parameters
    ----------
    p_params:
        Dict with keys: ``thinning``, ``point_size``, ``alpha``, ``elev``,
        ``azim``, ``zoom``, ``weights``, ``weights_title``.
    """
    data_arr = np.asarray(plot_field.array)
    n_plots  = data_arr.shape[0] if data_arr.ndim == 5 else 1
    ncols    = int(map_params["ncols"])
    nrows    = max(1, ceil(n_plots / ncols))

    titles = []
    for i in range(n_plots):
        t = (map_params["title_template"]
             .replace("%l%", selected_entry["label"])
             .replace("%i%", str(i)))
        t = _make_title(t, plot_field, i)
        titles.append(t)

    with _plt_lock:
        try:
            fig, _ = plot_field.plot(
                ncols=ncols,
                figsize=(float(map_params["fig_w"]) * ncols,
                         float(map_params["fig_h"]) * nrows),
                cmap=map_params["cmap"],
                colorbar=map_params["colorbar"],
                vmin=map_params["vmin"],
                vmax=map_params["vmax"],
                titles=titles,
                weights=p_params["weights"],
                weights_title=p_params["weights_title"],
                thinning=int(p_params["thinning"]),
                point_size=float(p_params["point_size"]),
                alpha=float(p_params["alpha"]),
                elev=float(p_params["elev"]),
                azim=float(p_params["azim"]),
                zoom=float(p_params["zoom"]),
            )
            png = _fig_to_png(fig, dpi=int(map_params["dpi"]))
        except Exception as e:
            print(f"[density_analysis] ParticleField map rendering failed: {e}")
            png = None
        finally:
            plt.close(fig)

    return png


# ---------------------------------------------------------------------------
# P(k) tab UI
# ---------------------------------------------------------------------------

def pk_tab(active_entries: list[dict], ref_field_type: str) -> None:
    """Render the full 3D P(k) tab."""
    from .utils import _DENSITY_3D

    if ref_field_type not in _DENSITY_3D:
        st.error(
            f"3D P(k) only supported for **DensityField**. "
            f"Current ref field type: **{ref_field_type}**. "
            "Use the **Angular Cl** tab for spherical/flat fields."
        )
        return

    spec_params_pk, spec_plot_pk = st.columns([1, 3])

    with spec_params_pk:
        with st.container(border=True):
            st.markdown("**Parameters**")

            pk_nl_fn = st.selectbox("Nonlinear fn", ["halofit", "linear"],
                                    key="analysis_pk_nl_fn")

            compare_theory_pk = st.checkbox("Compare against theory", value=False,
                                            key="analysis_pk_compare_theory")
            ratio_only_pk = st.checkbox("Ratio only (hide main panel)", value=False,
                                        key="analysis_ratio_only_pk",
                                        disabled=not compare_theory_pk,
                                        help="Show only the ratio panel without the main P(k) panel.")

            st.markdown("**Snapshot selection**")
            cached_pk = st.session_state.get("analysis_pk_results")
            if cached_pk:
                _ref_pk_arr = np.asarray(cached_pk[0][0][1].array)
                _ns_pk = _ref_pk_arr.shape[0] if _ref_pk_arr.ndim > 1 else 1
            else:
                _ns_pk = None
            _single_snap = (_ns_pk == 1)
            snap_index = st.text_input(
                "Snapshot index (numpy-style)", value=":",
                key="analysis_snap_index",
                disabled=_single_snap,
                help="Examples: ':' (all), '0:3', '-1:'. Selects which snapshots to plot.",
            )

            pk_fig_w   = st.number_input("Width/snapshot", min_value=2.0, max_value=16.0,
                                         value=5.0, step=0.5, key="analysis_pk_fig_w")
            pk_main_h  = st.number_input("Main panel height", min_value=1.0, max_value=10.0,
                                         value=3.0, step=0.5, key="analysis_pk_main_h")
            pk_ratio_h = st.number_input("Ratio panel height", min_value=0.5, max_value=5.0,
                                         value=1.0, step=0.25, key="analysis_pk_ratio_h")
            pk_dpi     = st.number_input("Render DPI", min_value=50, max_value=300,
                                         value=100, step=25, key="analysis_pk_dpi")

            st.markdown("**Shading bands** (set to 0 to disable)")
            pb1, pb2 = st.columns(2)
            with pb1:
                pk_band_10 = st.number_input("±10% band", min_value=0.0, max_value=100.0,
                                             value=10.0, step=1.0, key="analysis_pk_band_10")
            with pb2:
                pk_band_20 = st.number_input("±20% band", min_value=0.0, max_value=100.0,
                                             value=20.0, step=1.0, key="analysis_pk_band_20")
            pk_bands = [v / 100 for v in [pk_band_10, pk_band_20] if v > 0]

            has_pk = bool(st.session_state.get("analysis_pk_results"))
            pk_cb1, pk_cb2 = st.columns(2)
            with pk_cb1:
                pk_compute_btn = st.button("Compute", key="analysis_pk_compute_btn", type="primary")
            with pk_cb2:
                pk_redraw_btn  = st.button("Redraw", key="analysis_pk_redraw_btn",
                                           disabled=not has_pk,
                                           help="Re-render from cached P(k) values without recomputing")

    with spec_plot_pk:
        if pk_compute_btn:
            all_types_pk = {e["field_type"] for e in active_entries}
            if all_types_pk != _DENSITY_3D:
                st.error("3D P(k) only supported when all active fields are DensityField. "
                         f"Found: {', '.join(sorted(all_types_pk))}")
            else:
                pk_results = theory_pks = ref_fld_pk = None
                with st.spinner("Computing 3D power spectra..."):
                    pk_results, ref_fld_pk, ref_cosmo_pk = compute_pk(active_entries)
                if compare_theory_pk and pk_results:
                    with st.spinner("Computing theory P(k)..."):
                        theory_pks = compute_theory_pk(
                            ref_fld_pk, ref_cosmo_pk, pk_results, pk_nl_fn)
                else:
                    theory_pks = None
                st.session_state["analysis_pk_results"] = (pk_results, theory_pks, ref_fld_pk)

        pk_cached = st.session_state.get("analysis_pk_results")
        if (pk_compute_btn or pk_redraw_btn) and pk_cached:
            _pk_results, _theory_pks, _ref_fld_pk = pk_cached

            ref_pk_arr  = np.asarray(_pk_results[0][1].array)
            n_snaps_pk  = ref_pk_arr.shape[0] if ref_pk_arr.ndim > 1 else 1
            selected_snaps = parse_shell_index(
                snap_index if n_snaps_pk > 1 else ":", n_snaps_pk)

            if not selected_snaps:
                st.warning("No snapshots selected by the given index.")
            else:
                layout_params = {
                    "fig_w":   pk_fig_w,
                    "main_h":  pk_main_h,
                    "ratio_h": pk_ratio_h,
                }
                eff_theory    = compare_theory_pk and _theory_pks is not None
                eff_ratio_only = ratio_only_pk and eff_theory
                key     = (eff_theory, eff_ratio_only)
                builder = _PK_BUILDERS.get(key, _build_pk_main_only)

                with st.spinner("Rendering..."):
                    with _plt_lock:
                        fig_pk = builder(
                            _pk_results, _theory_pks,
                            selected_snaps, _ref_fld_pk,
                            layout_params, pk_bands,
                        )
                        st.session_state["analysis_pk_png"] = _fig_to_png(
                            fig_pk, dpi=int(pk_dpi))
                        st.session_state["analysis_pk_fig"] = fig_pk

        pk_png = st.session_state.get("analysis_pk_png")
        pk_fig = st.session_state.get("analysis_pk_fig")
        if pk_png:
            st.image(pk_png)
            if pk_fig is not None:
                from app.components.save_figure import render_save_figure
                render_save_figure(pk_fig, key_prefix="pk", filename="power_spectrum_3d")
        else:
            st.info("Click **Compute** to generate the 3D matter power spectrum.")
