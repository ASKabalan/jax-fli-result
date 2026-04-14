"""Shared utilities, constants, and plot helpers for the Analysis page."""
from __future__ import annotations

import io
from threading import RLock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Thread safety — matplotlib is not thread-safe
# ---------------------------------------------------------------------------
_plt_lock = RLock()

# ---------------------------------------------------------------------------
# Field type groups
# ---------------------------------------------------------------------------
_SPHERICAL_TYPES = {"SphericalDensity", "SphericalKappaField"}
_FLAT_TYPES = {"FlatDensity", "FlatKappaField"}
_KAPPA_TYPES = {"SphericalKappaField", "FlatKappaField"}
_DENSITY_3D = {"DensityField"}
_PARTICLE_TYPE = {"ParticleField"}
_SPECTRA_TYPES = {"PowerSpectrum"}

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
_COLOR_THEORY = "black"


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


def parse_slice(s: str) -> slice:
    """
    Parses strings like ':', '1:5', ':-1', or '-10:' into a slice object.
    """
    # Examples:
    # "1:10:2" -> slice(1, 10, 2)
    # "-10:"   -> slice(-10, None, None)
    # ":"      -> slice(None, None, None)
    s = s.strip()
    if not s or s == ":":
        return slice(None)

    # Split by ':' and take up to 3 parts (start, stop, step)
    parts = s.split(":")

    def to_int(val):
        val = val.strip()
        return int(val) if val else None

    try:
        # Pad parts to ensure we can unpack at least 1, up to 3
        # slice(*[None, None]) -> slice(None, None)
        return slice(*[to_int(p) for p in parts[:3]])
    except ValueError:
        # Fallback for invalid strings
        return slice(None)


# ---------------------------------------------------------------------------
# Figure utilities
# ---------------------------------------------------------------------------


def _fig_to_png(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _fig_to_pdf(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _make_title(template: str, field, idx: int) -> str:
    """Substitute %r%, %z%, %a% in a title template with field metadata."""
    title = template
    for attr, key, fmt in [
        ("comoving_centers", "%r%", ".1f"),
        ("z_sources", "%z%", ".3f"),
        ("scale_factors", "%a%", ".4f"),
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


def _add_shading(ax, bands: list[float]) -> None:
    """Draw ±pct axhspan bands (sorted descending, widest first) and a dotted 1.0 line."""
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    pcts = sorted(bands, reverse=True)
    base_alphas = [0.1, 0.25, 0.4]
    for i, frac in enumerate(pcts):
        alpha = base_alphas[i] if i < len(base_alphas) else 0.1
        ax.axhspan(1.0 - frac, 1.0 + frac, color="gray", alpha=alpha, zorder=0)


def _clean_ratio_ax(ax, ylabel: str, bands: list[float]) -> None:
    _add_shading(ax, bands)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yscale("linear")
    ax.set_ylim(0.85, 1.15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


# ---------------------------------------------------------------------------
# Pixel window
# ---------------------------------------------------------------------------


def pixel_window_function(ell, pixel_size_arcmin):
    """Pixel window function W_l = sinc²(l · θ_pix / 2π)."""
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180.0 * 60.0))
    return (np.sinc(ell * pixel_size_rad / (2 * np.pi))) ** 2


# ---------------------------------------------------------------------------
# JCAP publication style
# ---------------------------------------------------------------------------


def set_jcap_style() -> None:
    """Configure matplotlib for JCAP-quality publication figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 11,
            "figure.figsize": (6.0, 4.5),
            "figure.dpi": 150,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 6,
            "xtick.minor.size": 3,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
        }
    )
