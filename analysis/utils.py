import matplotlib.pyplot as plt

def set_jcap_style():
    plt.rcParams.update({
        # Font and LaTeX
        "text.usetex": True,           # Use LaTeX for all text
        "font.family": "serif",        # Standard JCAP serif font
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,               # Standard body text size
        
        # Figure Size (Single Column ~3.5 inches, Double Column ~7 inches)
        "figure.figsize": (6.0, 4.5),  # 4:3 ratio is generally safer for JCAP
        "figure.dpi": 150,             # High-res for screen, 300+ for export
        
        # Axes and Ticks
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,         # Standard line weight
        "xtick.top": True,             # Ticks on all sides
        "ytick.right": True,
        "xtick.direction": "in",       # Ticks pointing inward
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,       # Cleaner look without the box
        
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })

def generate_sample_cosmo_grid_catalog():
    from pathlib import Path
    import jax_fli as jfli
    import os

    os.makedirs("../results/sample_cosmogrid/catalog", exist_ok=True)
    
    # LC
    cosmogrid_path = Path("../../Simulations/CosmoGrid/raw/cosmo_000001/run_0/")
    catalog = jfli.io.load_cosmogrid_lc(cosmogrid_path, max_redshift=1.5, ud_nside=1024)

    num_shells = catalog.field[0].shape[0]
    # Save 20 shells per parquet file
    for i in range(0, num_shells, 20):
        shell_slice = slice(i, i + 20)
        output_path = f"../results/sample_cosmogrid/catalog/cosmogrid_sample_shells_{i}_{i+20}.parquet"
        catalog_subset = jfli.io.Catalog(
            cosmology=catalog.cosmology,
            field=[field[shell_slice] for field in catalog.fields],
        )
        catalog_subset.to_parquet(output_path)
        print(f"Saved {output_path} with shells {i} to {i+20}")

    # kappa

    stage3_path = "../../Simulations/CosmoGrid/stage3_forecast/cosmo_000002/perm_0000/"
    stage3_catalog = jfli.io.load_cosmogrid_kappa(stage3_path)
    stage3_catalog.to_parquet("../results/sample_cosmogrid/catalog/cosmogrid_sample_kappa.parquet")
    