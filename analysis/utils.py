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

