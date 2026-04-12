"""2D field map rendering for flat and spherical (healpy/flat-sky) field types.

All non-3D field types (FlatDensity, FlatKappaField, SphericalDensity,
SphericalKappaField) share the same rendering code path via plot_field.plot().
This module provides that shared renderer.

Angular Cl is NOT supported for flat types — the Cl tab shows an error for them.
"""
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from .utils import _fig_to_png, _make_title, _plt_lock


def render_field_map(
    selected_entry: dict,
    plot_field,
    map_params: dict,
) -> bytes | None:
    """Render a 2D (healpy or flat-sky) field map to PNG bytes.

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
    bytes or None
        PNG bytes on success; None if rendering failed (error is printed).
    """
    data_arr = np.asarray(plot_field.array)
    n_maps = 1 if data_arr.ndim <= 1 else int(np.prod(data_arr.shape[:-1]))
    ncols   = int(map_params["ncols"])
    nrows   = max(1, ceil(n_maps / ncols))

    titles = []
    for i in range(n_maps):
        t = (map_params["title_template"]
             .replace("%l%", selected_entry["label"])
             .replace("%i%", str(i)))
        t = _make_title(t, plot_field, i)
        titles.append(t)

    with _plt_lock:
        fig, axes = plt.subplots(
            nrows, ncols,
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
        except Exception as e:
            print(f"[flat_analysis] Field map rendering failed: {e}")
            png = None
        finally:
            plt.close(fig)

    return png
