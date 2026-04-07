import jax_fli as jfli
import jax.numpy as jnp
import numpy as np
import jax_cosmo as jc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches # For creating the shading legend boxes
import matplotlib.lines as mlines     # For checking line types in the legend
from matplotlib.ticker import FormatStrFormatter
import os
# --------------------------

# Load datasets
ref = jfli.io.Catalog.from_parquet('/home/wassim/Projects/NBody/jax-fli-result/results/01-step_size/KDK/h100_cosmo_M4096x4096x4096_B6000x6000x6000_STEPS30_c0.2589_S80.8159_s0.parquet')
compare = jfli.io.Catalog.from_parquet('/home/wassim/Projects/NBody/jax-fli-result/results/01-step_size/KDK/h100_cosmo_M4096x4096x4096_B6000x6000x6000_STEPS15_c0.2589_S80.8159_s0.parquet')
compare_2 = jfli.io.Catalog.from_parquet('/home/wassim/Projects/NBody/jax-fli-result/results/01-step_size/KDK/h100_cosmo_M4096x4096x4096_B6000x6000x6000_STEPS10_c0.2589_S80.8159_s0.parquet')

fld_ref = ref.field[0]
fld_compare = compare.field[0]
fld_compare_2 = compare_2.field[0]
cosmo = ref.cosmology[0]

LMAX = 1500
LMIN = 10
ells = jnp.arange(LMIN, LMAX + 1)
print(f"ells: {ells}")

print(f"Computing theory C_ell for density field with Halofit up to l={LMAX}...")
theory_result = jfli.compute_theory_cl_for_density(
                    cosmo, fld_ref, ells,
                    nonlinear_fn=jc.power.halofit, nz_zmax=float(1.5))

# Compute measured C_ell
fld_ref_od = fld_ref.to(jfli.units.OVERDENSITY, normalization="per_plane")
fld_compare_od = fld_compare.to(jfli.units.OVERDENSITY, normalization="per_plane")
fld_compare_2_od = fld_compare_2.to(jfli.units.OVERDENSITY, normalization="per_plane")

cl_ref = fld_ref_od.angular_cl(lmax=LMAX, method="healpy")[...,LMIN:]
cl_compare = fld_compare_od.angular_cl(lmax=LMAX, method="healpy")[...,LMIN:]
cl_compare_2 = fld_compare_2_od.angular_cl(lmax=LMAX, method="healpy")[...,LMIN:]

# Calculate Ratios
ratio_comp1_ref = cl_compare / cl_ref 
ratio_comp2_ref = cl_compare_2 / cl_ref 

ratio_ref_theory = cl_ref / theory_result
ratio_comp1_theory = cl_compare / theory_result
ratio_comp2_theory = cl_compare_2 / theory_result


# --- PLOT CONFIGURATION ---
COMPARE_AGAINST_REF = True
COMPARE_AGAINST_THEORY = True

# Shading Configuration
ENABLE_SHADING = True
SHADING_PERCENTAGES = [2.0, 5.0]  # E.g., [2.0, 5.0] draws ±2% and ±5% bands

# Define how many Cls to plot
NB_CL_TO_PLOT = 6

# Slice the arrays
cl_compare_to_plot = cl_compare[:NB_CL_TO_PLOT]
cl_compare_2_to_plot = cl_compare_2[:NB_CL_TO_PLOT]
cl_ref_to_plot = cl_ref[:NB_CL_TO_PLOT]
theory_result_to_plot = theory_result[:NB_CL_TO_PLOT]

ratio_comp1_ref_to_plot = ratio_comp1_ref[:NB_CL_TO_PLOT]
ratio_comp2_ref_to_plot = ratio_comp2_ref[:NB_CL_TO_PLOT]

ratio_ref_theory_to_plot = ratio_ref_theory[:NB_CL_TO_PLOT]
ratio_comp1_theory_to_plot = ratio_comp1_theory[:NB_CL_TO_PLOT]
ratio_comp2_theory_to_plot = ratio_comp2_theory[:NB_CL_TO_PLOT]

comoving_centers_to_plot = fld_ref_od.comoving_centers[:NB_CL_TO_PLOT]

# Consistent coloring
COLOR_THEORY = "black"
COLOR_REF = "tab:blue"
COLOR_COMP1 = "tab:orange"
COLOR_COMP2 = "tab:green"

def add_shading_zones(ax):
    """Adds a 1.0 reference line and dynamic shaded regions based on config."""
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    
    if ENABLE_SHADING and SHADING_PERCENTAGES:
        # Sort descending so the widest bands are drawn first (in the background)
        pcts = sorted(SHADING_PERCENTAGES, reverse=True)
        # Predefined alphas: widest band gets 0.1, tighter bands get darker
        base_alphas = [0.1, 0.25, 0.4] 
        
        for i, pct in enumerate(pcts):
            frac = pct / 100.0
            alpha = base_alphas[i] if i < len(base_alphas) else 0.1
            ax.axhspan(1.0 - frac, 1.0 + frac, color='gray', alpha=alpha, zorder=0, label=None)

def clean_ratio_axes(ax, ylabel):
    """Applies formatting to make ratio plots readable and avoid overlap."""
    add_shading_zones(ax)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yscale('linear') 
    ax.set_ylim(0.85, 1.15) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 

# --- Dynamic Staircase Coordinates Generator ---
coords = []
r = 0
while len(coords) < NB_CL_TO_PLOT:
    for c in range(r + 1):
        if len(coords) < NB_CL_TO_PLOT:
            coords.append((r, c))
    r += 1

# Calculate the required grid dimensions based on the generated coordinates
nrows = coords[-1][0] + 1
ncols = nrows # For a staircase, the grid is a square (N x N)

# --- Dynamic Plot Layout Setup ---
num_ratio_panels = sum([COMPARE_AGAINST_REF, COMPARE_AGAINST_THEORY])
if num_ratio_panels == 2:
    height_ratios = [2, 1, 1]
    figsize_y = 6 * nrows
elif num_ratio_panels == 1:
    height_ratios = [2, 1]
    figsize_y = 5 * nrows
else:
    height_ratios = [1]
    figsize_y = 4 * nrows

fig = plt.figure(figsize=(max(12, 6 * ncols), figsize_y)) 

# Adjust grid spacing
outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.08, hspace=0.15)

handles_out, labels_out = [], []

# Iterate over the specified subset
for i in range(NB_CL_TO_PLOT):
    row, col = coords[i]
    
    # Is this plot at the bottom of its current column?
    is_bottom = (row + 1, col) not in coords
    
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        num_ratio_panels + 1, 1,
        subplot_spec=outer_gs[row, col],
        height_ratios=height_ratios,
        hspace=0.05  
    )

    # -------------------------------------------------------------
    # 1. Main Plot: Cls 
    # -------------------------------------------------------------
    ax_main = fig.add_subplot(inner_gs[0, 0])
    
    theory_result_to_plot[i].plot(ax=ax_main, logx=True, logy=True, label="Theory", color=COLOR_THEORY, linestyle="--")
    cl_ref_to_plot[i].plot(ax=ax_main, logx=True, logy=True, label="STEPS30 (Ref)", color=COLOR_REF)
    cl_compare_to_plot[i].plot(ax=ax_main, logx=True, logy=True, label="STEPS15", color=COLOR_COMP1)
    cl_compare_2_to_plot[i].plot(ax=ax_main, logx=True, logy=True, label="STEPS5", color=COLOR_COMP2)
    
    ax_main.grid(True, which="both", ls="--", alpha=0.2)
    
    dist = float(comoving_centers_to_plot[i])
    ax_main.set_title(fr"$\chi = {dist:.2f}$ Mpc/h") 
    
    if col == 0:
        ax_main.set_ylabel(r"$C_\ell$")
    else:
        ax_main.tick_params(labelleft=False)
        
    ax_main.set_xlabel("") 
    
    if i == 0:
        handles_out, labels_out = ax_main.get_legend_handles_labels()
        
    if num_ratio_panels > 0:
        ax_main.tick_params(labelbottom=False)

    current_panel_idx = 1
    
    # -------------------------------------------------------------
    # 2. Ratio: Compare / Ref 
    # -------------------------------------------------------------
    if COMPARE_AGAINST_REF:
        ax_ratio_ref = fig.add_subplot(inner_gs[current_panel_idx, 0], sharex=ax_main)
        
        ratio_comp1_ref_to_plot[i].plot(ax=ax_ratio_ref, logx=True, color=COLOR_COMP1, legend=False)
        ratio_comp2_ref_to_plot[i].plot(ax=ax_ratio_ref, logx=True, color=COLOR_COMP2, legend=False)
        
        ylabel = "Ratio\n(vs Ref)" if col == 0 else ""
        clean_ratio_axes(ax_ratio_ref, ylabel)
        if col != 0:
            ax_ratio_ref.tick_params(labelleft=False)

        ax_ratio_ref.set_xlabel("")
        
        if current_panel_idx < num_ratio_panels:
            ax_ratio_ref.tick_params(labelbottom=False)
        else:
            if is_bottom:
                ax_ratio_ref.set_xlabel(r"$\ell$")
            else:
                ax_ratio_ref.tick_params(labelbottom=False)
            
        current_panel_idx += 1

    # -------------------------------------------------------------
    # 3. Ratio: All / Theory
    # -------------------------------------------------------------
    if COMPARE_AGAINST_THEORY:
        ax_ratio_th = fig.add_subplot(inner_gs[current_panel_idx, 0], sharex=ax_main)
        
        ratio_ref_theory_to_plot[i].plot(ax=ax_ratio_th, logx=True, color=COLOR_REF, legend=False)
        ratio_comp1_theory_to_plot[i].plot(ax=ax_ratio_th, logx=True, color=COLOR_COMP1, legend=False)
        ratio_comp2_theory_to_plot[i].plot(ax=ax_ratio_th, logx=True, color=COLOR_COMP2, legend=False)
        
        ylabel = "Ratio\n(vs Theory)" if col == 0 else ""
        clean_ratio_axes(ax_ratio_th, ylabel)
        if col != 0:
            ax_ratio_th.tick_params(labelleft=False)

        if is_bottom:
            ax_ratio_th.set_xlabel(r"$\ell$")
        else:
            ax_ratio_th.tick_params(labelbottom=False)

# --- ADD SHADING BANDS TO LEGEND ---
if ENABLE_SHADING and SHADING_PERCENTAGES:
    # 1. Add an invisible line to act as a spacer
    spacer = mlines.Line2D([], [], color='none')
    handles_out.append(spacer)
    labels_out.append('') # Blank text for the gap
    
    # 2. Add the actual patches
    pcts = sorted(SHADING_PERCENTAGES, reverse=True)
    base_alphas = [0.1, 0.25, 0.4]
    
    for i, pct in enumerate(pcts):
        alpha = base_alphas[i] if i < len(base_alphas) else 0.1
        # Create a filled rectangle representing the shading
        patch = mpatches.Patch(facecolor='gray', alpha=alpha, edgecolor='none')
        handles_out.append(patch)
        labels_out.append(f"±{pct}%")

# --- PLOT THE MASTER LEGEND OUTSIDE ---
# Positioned in the upper right empty space of the staircase
leg = fig.legend(
    handles_out, labels_out, 
    loc='upper right', 
    bbox_to_anchor=(0.9, 0.9), 
    fontsize=20, 
    title_fontsize=24, 
    title="", 
    frameon=True, 
    borderpad=1.5, 
    labelspacing=1.2
)

# Thicken ONLY the plotted lines, ignore the patches so they don't get heavy borders
for legobj in leg.legend_handles:
    if isinstance(legobj, mlines.Line2D):
        legobj.set_linewidth(6.0)

plt.savefig("cl_comparison.png", dpi=300, bbox_inches="tight")