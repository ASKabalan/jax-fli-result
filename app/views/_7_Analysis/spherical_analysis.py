"""SphericalDensity angular Cl analysis: computation, figure builders, and Cl tab UI.

Figure builders follow a strict "one function per plot mode" rule to keep each
case readable without nested ifs. All builders share the same signature so
dispatch is a single dictionary lookup with no argument-count branching.

Builders also used by kappa_spherical_analysis (imported from there).
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .utils import (
    _PALETTE, _COLOR_THEORY,
    _FLAT_TYPES,
    _fig_to_png, _make_title,
    _add_shading, _clean_ratio_ax,
    _plt_lock,
    parse_shell_index,
    pixel_window_function,
)


# ---------------------------------------------------------------------------
# Private layout helpers (also imported by kappa_spherical_analysis)
# ---------------------------------------------------------------------------

def _staircase_coords(n_selected: int) -> tuple[list[tuple[int, int]], int, int]:
    """Return (coords, nrows, ncols) for a lower-triangular staircase layout."""
    coords: list[tuple[int, int]] = []
    r = 0
    while len(coords) < n_selected:
        for c in range(r + 1):
            if len(coords) < n_selected:
                coords.append((r, c))
        r += 1
    nrows = coords[-1][0] + 1 if coords else 1
    return coords, nrows, nrows  # ncols == nrows for staircase


def _attach_legend(fig, handles, labels, bands: list[float]) -> None:
    """Append shading-band patches to existing handles and attach a figure legend."""
    if bands:
        spacer = mlines.Line2D([], [], color="none")
        handles.append(spacer)
        labels.append("")
        pcts_sorted = sorted(bands, reverse=True)
        base_alphas = [0.1, 0.25, 0.4]
        for i, frac in enumerate(pcts_sorted):
            alpha = base_alphas[i] if i < len(base_alphas) else 0.1
            patch = mpatches.Patch(facecolor="gray", alpha=alpha, edgecolor="none")
            handles.append(patch)
            labels.append(f"±{frac*100:.0f}%")

    leg = fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        fontsize=12,
        frameon=True,
        borderpad=1.0,
        labelspacing=0.8,
    )
    for legobj in leg.legend_handles:
        if isinstance(legobj, mlines.Line2D):
            legobj.set_linewidth(4.0)


def _make_inner_gs(outer_gs, row, col, height_ratios):
    """Create a nested GridSpec for one staircase cell."""
    return gridspec.GridSpecFromSubplotSpec(
        len(height_ratios), 1,
        subplot_spec=outer_gs[row, col],
        height_ratios=height_ratios,
        hspace=0.05,
    )


def _n_shells(spectra_results) -> int:
    arr = spectra_results[0][1].array
    return arr.shape[0] if arr.ndim > 1 else 1


def _cl_slice(cl, shell_idx: int, n_shells: int):
    return cl[shell_idx] if n_shells > 1 else cl


# ---------------------------------------------------------------------------
# Seven figure builders — one per (compare_ref, compare_theory, ratio_only)
# ---------------------------------------------------------------------------
# Uniform signature for all builders:
#   spectra_results  – list of (label, Cl)
#   theory_result    – Cl or None
#   selected_shells  – list[int] (shell indices to plot)
#   layout_params    – {"spec_fig_w", "spec_main_h", "spec_ratio_h"}
#   title_template   – str
#   field_ref        – field object for _make_title
#   bands            – list[float] (fractional shading bands, e.g. [0.02, 0.05])


def _build_cl_main_only(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Staircase Cl panels only — no ratio rows."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    height_ratios = [float(layout_params["spec_main_h"])]

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols),
                               height_ratios[0] * nrows))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)
        ax = fig.add_subplot(inner_gs[0, 0])

        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax, logx=True, logy=True, label=label, color=color)

        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.set_title(_make_title(title_template, field_ref, shell_idx))
        if col == 0:
            ax.set_ylabel(r"$C_\ell$")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(r"$\ell$")

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_with_ref_ratio(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Staircase Cl panels + ratio vs reference row."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    height_ratios = [float(layout_params["spec_main_h"]), float(layout_params["spec_ratio_h"])]
    figsize_y = sum(height_ratios) * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        # Main Cl panel
        ax_main = fig.add_subplot(inner_gs[0, 0])
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, field_ref, shell_idx))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        # Ratio vs Ref panel
        ax_r = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, shell_idx, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / ref_s).plot(ax=ax_r, logx=True, color=color, legend=False)
        ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        if is_bottom:
            ax_r.set_xlabel(r"$\ell$")
        else:
            ax_r.tick_params(labelbottom=False)

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_with_theory_ratio(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Staircase Cl panels + ratio vs theory row."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    height_ratios = [float(layout_params["spec_main_h"]), float(layout_params["spec_ratio_h"])]
    figsize_y = sum(height_ratios) * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        # Main Cl panel
        ax_main = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result
        th_s.plot(ax=ax_main, logx=True, logy=True,
                  label="Theory", color=_COLOR_THEORY, linestyle="--")
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, field_ref, shell_idx))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        # Ratio vs Theory panel
        ax_t = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_with_both_ratios(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Staircase Cl panels + ratio vs ref row + ratio vs theory row."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    height_ratios = [
        float(layout_params["spec_main_h"]),
        float(layout_params["spec_ratio_h"]),
        float(layout_params["spec_ratio_h"]),
    ]
    figsize_y = sum(height_ratios) * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        # Main Cl panel
        ax_main = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result
        th_s.plot(ax=ax_main, logx=True, logy=True,
                  label="Theory", color=_COLOR_THEORY, linestyle="--")
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)
        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, field_ref, shell_idx))
        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.tick_params(labelleft=False)
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        # Ratio vs Ref panel
        ax_r = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, shell_idx, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / ref_s).plot(ax=ax_r, logx=True, color=color, legend=False)
        ylabel_r = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel_r, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        ax_r.tick_params(labelbottom=False)
        ax_r.set_xlabel("")

        # Ratio vs Theory panel
        ax_t = fig.add_subplot(inner_gs[2, 0], sharex=ax_main)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel_t = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel_t, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_ratio_only_ref(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Ratio vs reference panels only — no main Cl panel."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    height_ratios = [float(layout_params["spec_ratio_h"])]
    figsize_y = height_ratios[0] * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax = fig.add_subplot(inner_gs[0, 0])
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, shell_idx, ns)
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / ref_s).plot(ax=ax, logx=True, color=color, legend=False, label=lbl)

        ax.set_title(_make_title(title_template, field_ref, shell_idx))
        ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\ell$")
        else:
            ax.tick_params(labelbottom=False)

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_ratio_only_theory(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Ratio vs theory panels only — no main Cl panel."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    height_ratios = [float(layout_params["spec_ratio_h"])]
    figsize_y = height_ratios[0] * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        ax = fig.add_subplot(inner_gs[0, 0])
        th_s = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / th_s).plot(ax=ax, logx=True, color=color, legend=False, label=lbl)

        ax.set_title(_make_title(title_template, field_ref, shell_idx))
        ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax, ylabel, bands)
        if col != 0:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\ell$")
        else:
            ax.tick_params(labelbottom=False)

        if i == 0:
            handles_out, labels_out = ax.get_legend_handles_labels()

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


def _build_cl_ratio_only_both(
    spectra_results, theory_result,
    selected_shells, layout_params, title_template, field_ref, bands,
) -> plt.Figure:
    """Both ratio rows (vs ref and vs theory) — no main Cl panel."""
    ns = _n_shells(spectra_results)
    n  = len(selected_shells)
    coords, nrows, ncols = _staircase_coords(n)
    coords_set = set(coords)
    rh = float(layout_params["spec_ratio_h"])
    height_ratios = [rh, rh]
    figsize_y = sum(height_ratios) * nrows

    fig = plt.figure(figsize=(max(12, float(layout_params["spec_fig_w"]) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []
    for i, (row, col) in enumerate(coords):
        shell_idx = selected_shells[i]
        is_bottom = (row + 1, col) not in coords_set
        inner_gs  = _make_inner_gs(outer_gs, row, col, height_ratios)

        th_s  = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result
        _, ref_cl = spectra_results[0]
        ref_s = _cl_slice(ref_cl, shell_idx, ns)

        # Ratio vs Ref
        ax_r = fig.add_subplot(inner_gs[0, 0])
        for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / ref_s).plot(ax=ax_r, logx=True, color=color, legend=False, label=lbl)
        ax_r.set_title(_make_title(title_template, field_ref, shell_idx))
        ylabel_r = "Ratio\n(vs Ref)" if col == 0 else ""
        _clean_ratio_ax(ax_r, ylabel_r, bands)
        if col != 0:
            ax_r.tick_params(labelleft=False)
        ax_r.tick_params(labelbottom=False)
        ax_r.set_xlabel("")
        if i == 0:
            handles_out, labels_out = ax_r.get_legend_handles_labels()

        # Ratio vs Theory
        ax_t = fig.add_subplot(inner_gs[1, 0], sharex=ax_r)
        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s  = _cl_slice(cl, shell_idx, ns)
            (cl_s / th_s).plot(ax=ax_t, logx=True, color=color, legend=False)
        ylabel_t = "Ratio\n(vs Theory)" if col == 0 else ""
        _clean_ratio_ax(ax_t, ylabel_t, bands)
        if col != 0:
            ax_t.tick_params(labelleft=False)
        if is_bottom:
            ax_t.set_xlabel(r"$\ell$")
        else:
            ax_t.tick_params(labelbottom=False)

    _attach_legend(fig, handles_out, labels_out, bands)
    return fig


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_CL_BUILDERS = {
    (False, False, False): _build_cl_main_only,
    (True,  False, False): _build_cl_with_ref_ratio,
    (False, True,  False): _build_cl_with_theory_ratio,
    (True,  True,  False): _build_cl_with_both_ratios,
    (True,  False, True):  _build_cl_ratio_only_ref,
    (False, True,  True):  _build_cl_ratio_only_theory,
    (True,  True,  True):  _build_cl_ratio_only_both,
}


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def compute_cls(
    active_entries: list[dict],
    lmin: int,
    lmax: int,
) -> list[tuple[str, object]]:
    """Compute angular Cl for SphericalDensity entries.

    Density fields are normalised (divided by mean, minus 1) before computing.
    """
    spectra_results = []
    for entry in active_entries:
        fld = entry["catalog"].field[0]
        ft  = entry["field_type"]
        if ft in ("SphericalDensity", "FlatDensity"):
            fld = (fld / fld.array.mean(axis=-1, keepdims=True)) - 1.0
        if ft in _FLAT_TYPES:
            _n_ell_bins = min(100, int(lmax) - int(lmin) + 1)
            _ell_edges  = np.linspace(int(lmin), int(lmax), _n_ell_bins + 1)
            cl = fld.angular_cl(ell_edges=_ell_edges)
        else:
            cl = fld.angular_cl(lmax=int(lmax), method="healpy")[..., int(lmin):]
        spectra_results.append((entry["label"], cl))
    return spectra_results


def compute_theory_cl(
    ref_obj,
    ells,
    nl_fn_name: str,
    nz_zmax: float,
    apply_pixwin: bool,
    lmin: int,
    lmax: int,
) -> object | None:
    """Compute theory Cl for SphericalDensity fields.

    Optionally divides theory by the healpy pixel window function squared.
    """
    import jax_cosmo as jc
    import jax_fli as jfli

    ref_field_obj = ref_obj.field[0]
    ref_cosmo     = ref_obj.cosmology[0]
    nl_fn = jc.power.halofit if nl_fn_name == "halofit" else "linear"

    theory_result = jfli.compute_theory_cl_for_density(
        ref_cosmo, ref_field_obj, ells,
        nonlinear_fn=nl_fn, nz_zmax=float(nz_zmax),
    )

    if apply_pixwin and theory_result is not None:
        ft = type(ref_field_obj).__name__
        if ft not in _FLAT_TYPES:
            # Spherical: use healpy pixel window
            import healpy as hp
            nside_val = int(ref_field_obj.nside)
            pw_full   = np.asarray(hp.pixwin(nside_val, lmax=int(lmax)))
            pw        = pw_full[int(lmin):]
            new_arr   = theory_result.array / pw**2
            theory_result = theory_result.replace(array=new_arr)
        else:
            # Flat: sinc² pixel window
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
# Cl tab UI
# ---------------------------------------------------------------------------

def cl_tab(active_entries: list[dict], ref_field_type: str) -> None:
    """Render the full Angular Cl tab for SphericalDensity (and FlatDensity)."""
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
            nz_zmax = st.number_input("nz_zmax", min_value=0.1, value=2.0,
                                      step=0.1, format="%.1f", key="analysis_nz_zmax")

            compare_ref = st.checkbox("Compare against reference", value=False,
                                      key="analysis_compare_ref",
                                      disabled=(len(active_entries) < 2))
            compare_theory = st.checkbox("Compare against theory", value=False,
                                          key="analysis_compare_theory")
            ratio_only = st.checkbox("Ratio only (hide main panel)", value=False,
                                     key="analysis_ratio_only_cl",
                                     disabled=not (compare_ref or compare_theory),
                                     help="Show only ratio panels without the main Cl panel.")

            # Pixel window correction
            apply_pixwin = False
            if compare_theory:
                apply_pixwin = st.checkbox(
                    "Divide theory by pixel window", value=False,
                    key="analysis_pixwin",
                    help="For spherical: uses healpy.pixwin(nside). "
                         "For flat: uses sinc² pixel window from pixel scale.",
                )

            st.markdown("**Shell selection**")
            # Determine n_shells from cached results if available, else show enabled
            cached = st.session_state.get("analysis_spectra_results")
            if cached:
                _ns = cached[0][1].array.shape[0] if cached[0][1].array.ndim > 1 else 1
            else:
                _ns = None  # unknown until computed
            _single_shell = (_ns == 1)
            cl_index = st.text_input(
                "Shell index (numpy-style)", value=":",
                key="analysis_cl_index",
                disabled=_single_shell,
                help="Examples: ':' (all), '0:6', '::2', '-3:'. "
                     "Disabled when there is only one shell.",
            )

            st.markdown("**Plot layout**")
            spec_fig_w  = st.number_input("Width/col", min_value=2.0, max_value=16.0,
                                          value=5.0, step=0.5, key="analysis_spec_fig_w")
            spec_main_h = st.number_input("Main panel height", min_value=1.0, max_value=10.0,
                                          value=3.0, step=0.5, key="analysis_spec_main_h")
            spec_ratio_h = st.number_input("Ratio panel height", min_value=0.5, max_value=5.0,
                                           value=1.0, step=0.25, key="analysis_spec_ratio_h")
            spec_dpi = st.number_input("Render DPI", min_value=50, max_value=300,
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
        # --- Computation (slow JAX step) — only on Compute click ---
        if compute_btn:
            bad_entries = [e["field_type"] for e in active_entries
                           if e["field_type"] == "ParticleField"]
            if bad_entries:
                st.error("Spectra plotting not supported for ParticleField type.")
            else:
                spectra_results = []
                theory_result   = None

                with st.spinner("Computing angular power spectra..."):
                    spectra_results = compute_cls(active_entries, int(lmin), int(lmax))

                if compare_theory and spectra_results:
                    ref_cl_wavenumber = spectra_results[0][1].wavenumber
                    ells = jnp.asarray(ref_cl_wavenumber)
                    with st.spinner("Computing theory Cl..."):
                        theory_result = compute_theory_cl(
                            active_entries[0]["catalog"],
                            ells, nonlinear_fn_name,
                            float(nz_zmax), apply_pixwin,
                            int(lmin), int(lmax),
                        )

                st.session_state["analysis_spectra_results"] = spectra_results
                st.session_state["analysis_theory_result"]   = theory_result

        # --- Rendering (matplotlib step) — on Compute or Redraw click ---
        spectra_results = st.session_state.get("analysis_spectra_results")
        theory_result   = st.session_state.get("analysis_theory_result")

        if (compute_btn or redraw_btn) and spectra_results:
            ns = spectra_results[0][1].array.shape[0] if spectra_results[0][1].array.ndim > 1 else 1
            selected_shells = parse_shell_index(cl_index if ns > 1 else ":", ns)
            if not selected_shells:
                st.warning("No shells selected by the given index.")
            else:
                layout_params = {
                    "spec_fig_w":  spec_fig_w,
                    "spec_main_h": spec_main_h,
                    "spec_ratio_h": spec_ratio_h,
                }
                eff_compare_ref    = compare_ref and len(spectra_results) > 1
                eff_compare_theory = compare_theory and theory_result is not None
                eff_ratio_only     = ratio_only and (eff_compare_ref or eff_compare_theory)
                key = (eff_compare_ref, eff_compare_theory, eff_ratio_only)
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

        # --- Display persisted PNG ---
        spectra_png = st.session_state.get("analysis_spectra_png")
        spectra_fig = st.session_state.get("analysis_spectra_fig")
        if spectra_png:
            st.image(spectra_png)
            if spectra_fig is not None:
                from app.components.save_figure import render_save_figure
                render_save_figure(spectra_fig, key_prefix="spectra", filename="spectra")
        else:
            st.info("Click **Compute** to generate angular power spectra.")
