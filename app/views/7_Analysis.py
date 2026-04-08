"""Analysis page — load parquet files, visualize maps, compare angular power spectra."""
import io
import os
from math import ceil
from pathlib import Path
from threading import RLock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import FormatStrFormatter

from app.components.styled_container import inject_custom_css

inject_custom_css()

# Matplotlib is not thread-safe — serialise all figure creation/save with this lock.
_plt_lock = RLock()

st.title("Analysis")

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
CATALOGS_KEY = "analysis_catalogs"

if CATALOGS_KEY not in st.session_state:
    st.session_state[CATALOGS_KEY] = []


def _load_catalog(path: str):
    import jax_fli as jfli
    path = path.strip()
    if not path:
        return
    for entry in st.session_state[CATALOGS_KEY]:
        if entry["path"] == path:
            st.toast(f"Already loaded: {Path(path).name}")
            return
    try:
        catalog = jfli.io.Catalog.from_parquet(path)
        field_type = type(catalog.field[0]).__name__
        st.session_state[CATALOGS_KEY].append({
            "path": path,
            "label": Path(path).name,
            "catalog": catalog,
            "field_type": field_type,
            "active": True,
        })
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")


def _toggle_active(index: int, key: str):
    st.session_state[CATALOGS_KEY][index]["active"] = st.session_state[key]


def _remove_catalog(index: int):
    st.session_state[CATALOGS_KEY].pop(index)


def _move_up(index: int):
    entries = st.session_state[CATALOGS_KEY]
    entries[index - 1], entries[index] = entries[index], entries[index - 1]


def _move_down(index: int):
    entries = st.session_state[CATALOGS_KEY]
    entries[index], entries[index + 1] = entries[index + 1], entries[index]


def _update_label(index: int, key: str):
    st.session_state[CATALOGS_KEY][index]["label"] = st.session_state[key]



def _fig_to_png(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _make_title(template: str, field, idx: int) -> str:
    """Substitute %r%, %z%, %a% in the title template with field metadata."""
    title = template
    for attr, key, fmt in [
        ("comoving_centers", "%r%", ".1f"),
        ("z_sources",        "%z%", ".3f"),
        ("scale_factors",    "%a%", ".4f"),
    ]:
        if key not in title:
            continue
        arr = getattr(field, attr, None)
        if arr is not None:
            try:
                title = title.replace(key, format(float(np.asarray(arr)[idx]), fmt))
            except Exception:
                pass
    return title


_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
_COLOR_THEORY = "black"


def _add_shading(ax, bands: list[float]):
    """Draw ±pct axhspan bands (sorted descending, widest first) and a 1.0 dotted line."""
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    pcts = sorted(bands, reverse=True)
    base_alphas = [0.1, 0.25, 0.4]
    for i, frac in enumerate(pcts):
        alpha = base_alphas[i] if i < len(base_alphas) else 0.1
        ax.axhspan(1.0 - frac, 1.0 + frac, color="gray", alpha=alpha, zorder=0)


def _clean_ratio_ax(ax, ylabel: str, bands: list[float]):
    _add_shading(ax, bands)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yscale("linear")
    ax.set_ylim(0.85, 1.15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _build_spectra_fig(
    spectra_results, theory_result,
    compare_ref: bool, compare_theory: bool,
    nb_cl: int,
    spec_fig_w: float,
    spec_main_h: float, spec_ratio_h: float,
    bands: list[float],
    title_template: str,
    field_ref,
):
    """Create and return the spectra Figure (caller must close). Returns None if no data."""
    n_shells = (spectra_results[0][1].array.shape[0]
                if spectra_results[0][1].array.ndim > 1 else 1)
    n_selected = min(nb_cl, n_shells)
    if n_selected == 0:
        return None

    has_ref_panel = compare_ref and len(spectra_results) > 1
    has_theory_panel = compare_theory and theory_result is not None
    n_ratio_panels = int(has_ref_panel) + int(has_theory_panel)

    # Height ratios per cell
    if n_ratio_panels == 2:
        height_ratios = [spec_main_h, spec_ratio_h, spec_ratio_h]
    elif n_ratio_panels == 1:
        height_ratios = [spec_main_h, spec_ratio_h]
    else:
        height_ratios = [spec_main_h]

    # Staircase coordinates (mirrors script_compare.py)
    coords = []
    r = 0
    while len(coords) < n_selected:
        for c in range(r + 1):
            if len(coords) < n_selected:
                coords.append((r, c))
        r += 1
    nrows = coords[-1][0] + 1
    ncols = nrows
    coords_set = set(coords)

    figsize_y = sum(height_ratios) * nrows
    fig = plt.figure(figsize=(max(12, float(spec_fig_w) * ncols), figsize_y))
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.25)

    handles_out, labels_out = [], []

    for i, (row, col) in enumerate(coords):
        shell_idx = i  # simple [:nb_cl] slice
        is_bottom = (row + 1, col) not in coords_set

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_ratio_panels + 1, 1,
            subplot_spec=outer_gs[row, col],
            height_ratios=height_ratios,
            hspace=0.05,
        )

        # --- Main Cl panel ---
        ax_main = fig.add_subplot(inner_gs[0, 0])

        if has_theory_panel:
            th_s = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result
            th_s.plot(ax=ax_main, logx=True, logy=True,
                      label="Theory", color=_COLOR_THEORY, linestyle="--")

        for ci, (lbl, cl) in enumerate(spectra_results):
            color = _PALETTE[ci % len(_PALETTE)]
            cl_s = cl[shell_idx] if n_shells > 1 else cl
            label = f"{lbl} (Ref)" if ci == 0 else lbl
            cl_s.plot(ax=ax_main, logx=True, logy=True, label=label, color=color)

        ax_main.grid(True, which="both", ls="--", alpha=0.2)
        ax_main.set_title(_make_title(title_template, field_ref, shell_idx))

        if col == 0:
            ax_main.set_ylabel(r"$C_\ell$")
        else:
            ax_main.tick_params(labelleft=False)
        ax_main.set_xlabel("")

        if i == 0:
            handles_out, labels_out = ax_main.get_legend_handles_labels()

        if n_ratio_panels > 0:
            ax_main.tick_params(labelbottom=False)

        current_panel_idx = 1

        # --- Ratio vs Ref panel ---
        if has_ref_panel:
            ax_r = fig.add_subplot(inner_gs[current_panel_idx, 0], sharex=ax_main)
            _, ref_cl = spectra_results[0]
            ref_s = ref_cl[shell_idx] if n_shells > 1 else ref_cl

            for ci, (lbl, cl) in enumerate(spectra_results[1:], 1):
                color = _PALETTE[ci % len(_PALETTE)]
                cl_s = cl[shell_idx] if n_shells > 1 else cl
                ratio = cl_s / ref_s
                ratio.plot(ax=ax_r, logx=True, color=color, legend=False)

            ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
            _clean_ratio_ax(ax_r, ylabel, bands)
            if col != 0:
                ax_r.tick_params(labelleft=False)
            ax_r.set_xlabel("")

            if current_panel_idx < n_ratio_panels:
                ax_r.tick_params(labelbottom=False)
            else:
                if is_bottom:
                    ax_r.set_xlabel(r"$\ell$")
                else:
                    ax_r.tick_params(labelbottom=False)
            current_panel_idx += 1

        # --- Ratio vs Theory panel ---
        if has_theory_panel:
            ax_t = fig.add_subplot(inner_gs[current_panel_idx, 0], sharex=ax_main)
            th_s = theory_result[shell_idx] if theory_result.array.ndim > 1 else theory_result

            for ci, (lbl, cl) in enumerate(spectra_results):
                color = _PALETTE[ci % len(_PALETTE)]
                cl_s = cl[shell_idx] if n_shells > 1 else cl
                ratio = cl_s / th_s
                ratio.plot(ax=ax_t, logx=True, color=color, legend=False)

            ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
            _clean_ratio_ax(ax_t, ylabel, bands)
            if col != 0:
                ax_t.tick_params(labelleft=False)

            if is_bottom:
                ax_t.set_xlabel(r"$\ell$")
            else:
                ax_t.tick_params(labelbottom=False)

    # --- Master legend (shading patches + line handles) ---
    if bands:
        spacer = mlines.Line2D([], [], color="none")
        handles_out.append(spacer)
        labels_out.append("")
        pcts_sorted = sorted(bands, reverse=True)
        base_alphas = [0.1, 0.25, 0.4]
        for i, frac in enumerate(pcts_sorted):
            alpha = base_alphas[i] if i < len(base_alphas) else 0.1
            patch = mpatches.Patch(facecolor="gray", alpha=alpha, edgecolor="none")
            handles_out.append(patch)
            labels_out.append(f"±{frac*100:.0f}%")

    leg = fig.legend(
        handles_out, labels_out,
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

    return fig


# ---------------------------------------------------------------------------
# Section 1: File Loading
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Load Parquet Files")
    col_path, col_btn = st.columns([5, 1])
    with col_path:
        new_path = st.text_input("Parquet file path", key="analysis_new_path",
                                 placeholder="/path/to/file.parquet",
                                 label_visibility="collapsed")
    with col_btn:
        st.button("Load", key="analysis_load_btn", on_click=_load_catalog, args=(new_path,))

    entries = st.session_state[CATALOGS_KEY]
    for i, entry in enumerate(entries):
        cb, cl, ct, cu, cd, cr = st.columns([0.5, 4, 2, 0.5, 0.5, 0.5])
        with cb:
            st.checkbox(f"**{'REF' if i == 0 else f'#{i+1}'}**",
                        value=entry.get("active", True),
                        key=f"analysis_active_{i}",
                        on_change=_toggle_active, args=(i, f"analysis_active_{i}"))
        with cl:
            st.text_input("Label", value=entry["label"], key=f"analysis_label_{i}",
                          on_change=_update_label, args=(i, f"analysis_label_{i}"),
                          label_visibility="collapsed")
        with ct:
            st.caption(entry["field_type"])
        with cu:
            st.button("\u2191", key=f"analysis_up_{i}", on_click=_move_up,
                      args=(i,), disabled=(i == 0))
        with cd:
            st.button("\u2193", key=f"analysis_down_{i}", on_click=_move_down,
                      args=(i,), disabled=(i == len(entries) - 1))
        with cr:
            st.button("\u2716", key=f"analysis_rm_{i}", on_click=_remove_catalog, args=(i,))

    if not entries:
        st.info("No files loaded. Enter a parquet file path and click Load.")

if not entries:
    st.stop()

# ---------------------------------------------------------------------------
# Section 2: Field Map Visualization
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Visualization")

file_labels = [f"[{'REF' if i == 0 else i+1}] {e['label']}" for i, e in enumerate(entries)]
selected_idx = st.selectbox("Select file to display", range(len(entries)),
                             format_func=lambda i: file_labels[i],
                             key="analysis_vis_select")

selected_entry = entries[selected_idx]
field = selected_entry["catalog"].field[0]

# Invalidate cached PNG when the selected file changes.
_fkey = selected_entry["path"]
if st.session_state.get("_field_cache_path") != _fkey:
    st.session_state.pop("analysis_field_png", None)
    st.session_state["_field_cache_path"] = _fkey

# Plot settings + Plot button
with st.container(border=True):
    st.markdown("**Field Map Settings**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        map_index = st.text_input("Index (numpy-like)", value=":", key="analysis_map_index")
        map_ncols = st.number_input("Columns", min_value=1, max_value=10, value=2,
                                    key="analysis_map_ncols")
        map_projection = st.selectbox("Projection",
                                      ["mollweide", "cart", "polar", "aitoff", "hammer", "lambert"],
                                      key="analysis_map_proj")
    with mc2:
        map_cmap = st.selectbox("Colormap",
                                ["magma", "viridis", "inferno", "plasma", "cividis",
                                 "coolwarm", "RdBu_r", "hot", "bone", "gray"],
                                key="analysis_map_cmap")
        map_border = st.number_input("Border width", min_value=0.0, max_value=5.0,
                                     value=1.0, step=0.5, key="analysis_map_border")
    with mc3:
        map_fig_w = st.number_input("Width/map", min_value=2.0, max_value=12.0,
                                    value=4.0, step=0.5, key="analysis_map_fig_w")
        map_fig_h = st.number_input("Height/map", min_value=2.0, max_value=12.0,
                                    value=4.0, step=0.5, key="analysis_map_fig_h")
    with mc4:
        map_colorbar = st.checkbox("Colorbar", value=True, key="analysis_map_cbar")
        map_ticks = st.checkbox("Graticule", value=False, key="analysis_map_ticks")
    with mc5:
        map_use_vmin = st.checkbox("Custom vmin", value=False, key="analysis_map_use_vmin")
        map_vmin = st.number_input("vmin", value=0.0, format="%.4f",
                                   key="analysis_map_vmin", disabled=not map_use_vmin)
        map_use_vmax = st.checkbox("Custom vmax", value=False, key="analysis_map_use_vmax")
        map_vmax = st.number_input("vmax", value=1.0, format="%.4f",
                                   key="analysis_map_vmax", disabled=not map_use_vmax)

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        map_title_template = st.text_input(
            "Panel title template", value="%l% - %i%",
            key="analysis_map_title_template",
            help="%l% = Label | %i% = Index | %r% = comoving distance  |  %z% = redshift  |  %a% = scale factor"
        )
    with tc2:
        map_dpi = st.number_input("Render DPI", min_value=50, max_value=300, value=100,
                                  step=25, key="analysis_map_dpi")
    plot_btn = st.button("Plot", key="analysis_plot_btn", type="primary")

# Map display — only renders on Plot click; result persists as PNG bytes.
with st.container(border=True):
    st.markdown("**Field Map**")

    if plot_btn:
        try:
            idx = eval(f"np.s_[{map_index}]", {"np": np})
            plot_field = field[idx]
        except Exception as e:
            st.error(f"Invalid index '{map_index}': {e}")
            plot_field = field

        data_arr = np.asarray(plot_field.array)
        n_maps = 1 if data_arr.ndim <= 1 else int(np.prod(data_arr.shape[:-1]))
        nrows = max(1, ceil(n_maps / int(map_ncols)))
        with _plt_lock:
            fig_map, axes_map = plt.subplots(nrows, int(map_ncols),
                                             figsize=(map_fig_w * map_ncols,
                                                      map_fig_h * nrows))
            try:
                titles = []
                for i in range(n_maps):
                    t = map_title_template.replace("%l%", selected_entry["label"]).replace("%i%", str(i))
                    t = _make_title(t, plot_field, i)
                    titles.append(t)
                plot_field.plot(ax=axes_map,
                           titles=titles,
                           border_linewidth=map_border,
                           cmap=map_cmap,
                           colorbar=map_colorbar,
                           show_ticks=map_ticks,
                           projection_type=map_projection,
                           vmin=map_vmin if map_use_vmin else None,
                           vmax=map_vmax if map_use_vmax else None)
                st.session_state["analysis_field_png"] = _fig_to_png(fig_map, dpi=int(map_dpi))
            except Exception as e:
                st.error(f"Field map rendering failed: {e}")
            finally:
                plt.close(fig_map)

    field_png = st.session_state.get("analysis_field_png")
    if field_png:
        st.image(field_png)
        st.download_button("Download PNG", data=field_png,
                           file_name="field_map.png", mime="image/png",
                           key="save_field_dl")
    else:
        st.info("Adjust settings above, then click **Plot**.")

# Metadata mini-plots (always rendered — fast, no healpy involved).
meta_attrs = [
    (a, lbl, u)
    for a, lbl, u in [
        ("comoving_centers", "Comoving Centers", "Mpc/h"),
        ("scale_factors",    "Scale Factors",    "a"),
        ("z_sources",        "z Sources",        "z"),
        ("density_width",    "Density Width",    "Mpc/h"),
    ]
    if getattr(field, a, None) is not None
]

if meta_attrs:
    cols = st.columns(len(meta_attrs))
    for col, (attr, lbl, unit) in zip(cols, meta_attrs):
        with col:
            arr = np.asarray(getattr(field, attr))
            with _plt_lock:
                fig_m, ax_m = plt.subplots(figsize=(4, 2.5))
                if arr.ndim == 0:
                    ax_m.axhline(float(arr), color="C0")
                else:
                    ax_m.plot(arr, marker="o", markersize=3)
                    ax_m.set_xticks(np.arange(len(arr)))
                    ax_m.grid(True, linestyle="--", alpha=0.7)
                ax_m.set_title(lbl, fontsize=10)
                ax_m.set_xlabel("Shell" if arr.ndim > 0 else "")
                ax_m.set_ylabel(unit, fontsize=9)
                ax_m.tick_params(labelsize=8)
                fig_m.tight_layout()
                st.pyplot(fig_m)
                plt.close(fig_m)

with st.expander("Field info"):
    st.code(repr(field), language=None)

# ---------------------------------------------------------------------------
# Section 3: Spectra Comparison
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Angular Power Spectra")

spec_params, spec_plot = st.columns([1, 3])

with spec_params:
    with st.container(border=True):
        st.markdown("**Parameters**")

        active_entries = [e for e in entries if e.get("active", True)]
        if not active_entries:
            st.warning("No active entries selected.")
            st.stop()

        default_nside = getattr(active_entries[0]["catalog"].field[0], "nside", 512)
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
                                  key="analysis_compare_ref", disabled=(len(active_entries) < 2))
        compare_theory = st.checkbox("Compare against theory", value=False,
                                     key="analysis_compare_theory")

        ref_field_type = active_entries[0]["field_type"]
        nz_shear_mode = None
        if ref_field_type == "SphericalKappaField" and compare_theory:
            st.markdown("**Kappa settings**")
            nz_shear_mode = st.radio("nz_shear", ["s3", "point sources"],
                                     key="analysis_nz_shear_mode")

        st.markdown("**Shell selection**")
        nb_cl = st.number_input("Number of Cls to plot", min_value=1, max_value=200, value=6,
                                key="analysis_nb_cl",
                                help="Plots the first N shells (top N by index)")

        st.markdown("**Plot layout**")
        spec_fig_w = st.number_input("Width/col", min_value=2.0, max_value=16.0,
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
            help="%r% = comoving distance  |  %z% = redshift  |  %a% = scale factor"
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
            redraw_btn = st.button("Redraw", key="analysis_redraw_btn",
                                   disabled=not has_spectra,
                                   help="Re-render from cached Cl values without recomputing")

with spec_plot:
    # --- Computation (slow JAX step) — only on Compute click ---
    if compute_btn:
        import jax.numpy as jnp
        import jax_cosmo as jc

        ells = jnp.arange(int(lmin), int(lmax) + 1)
        spectra_results = []
        theory_result = None

        with st.spinner("Computing angular power spectra..."):
            for entry in active_entries:
                fld = entry["catalog"].field[0]
                if entry["field_type"] in ("SphericalDensity", "FlatDensity"):
                    fld = (fld / fld.array.mean(axis=-1, keepdims=True)) - 1.0
                cl = fld.angular_cl(lmax=int(lmax), method="healpy")[..., int(lmin):]
                spectra_results.append((entry["label"], cl))

        if compare_theory:
            with st.spinner("Computing theory Cl..."):
                import jax_fli as jfli
                ref_obj = active_entries[0]["catalog"]
                ref_field_obj = ref_obj.field[0]
                ref_cosmo = ref_obj.cosmology[0]
                nl_fn = jc.power.halofit if nonlinear_fn_name == "halofit" else "linear"
                if ref_field_type in ("SphericalDensity", "FlatDensity"):
                    theory_result = jfli.compute_theory_cl_for_density(
                        ref_cosmo, ref_field_obj, ells,
                        nonlinear_fn=nl_fn, nz_zmax=float(nz_zmax))
                elif ref_field_type == "SphericalKappaField":
                    z_src = (jfli.io.get_stage3_nz_shear() if nz_shear_mode == "s3"
                             else ref_field_obj.z_sources.tolist())
                    theory_result = jfli.compute_theory_cl(
                        ref_cosmo, ells, z_src,
                        probe_type="weak_lensing", nonlinear_fn=nl_fn)

        st.session_state["analysis_spectra_results"] = spectra_results
        st.session_state["analysis_theory_result"] = theory_result

    # --- Rendering (matplotlib step) — on Compute or Redraw click ---
    spectra_results = st.session_state.get("analysis_spectra_results")
    theory_result = st.session_state.get("analysis_theory_result")

    if (compute_btn or redraw_btn) and spectra_results:
        with st.spinner("Rendering..."):
            with _plt_lock:
                fig_cl = _build_spectra_fig(
                    spectra_results, theory_result,
                    compare_ref, compare_theory,
                    int(nb_cl),
                    spec_fig_w, spec_main_h, spec_ratio_h, bands,
                    title_template,
                    active_entries[0]["catalog"].field[0],
                )
                if fig_cl is not None:
                    old_fig = st.session_state.pop("analysis_spectra_fig", None)
                    if old_fig is not None:
                        plt.close(old_fig)
                    st.session_state["analysis_spectra_png"] = _fig_to_png(
                        fig_cl, dpi=int(spec_dpi))
                    st.session_state["analysis_spectra_fig"] = fig_cl
                else:
                    st.warning("No valid shell indices selected.")
                    st.session_state.pop("analysis_spectra_png", None)

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
