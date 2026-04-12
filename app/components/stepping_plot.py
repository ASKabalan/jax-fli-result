"""Stepping plot component — visualises simulation time-stepping geometry."""
from __future__ import annotations

from threading import RLock

import streamlit as st

# RLock for any future matplotlib usage in this module; plotly itself is thread-safe.
_plt_lock = RLock()

_PLOT_KEY = "stepping_plot_fig"


@st.cache_resource
def _load_jax_modules():
    """Lazy-load heavy JAX dependencies once."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    import jax_cosmo as jc
    from jax_fli.pm._resolve_geometry import simulation_stepping
    return jc, simulation_stepping


@st.cache_data
def _compute_stepping(
    t0: float,
    t1: float,
    nb_steps: int,
    ts: tuple[float, ...],   # snapshot scale factors — replaces nb_shells
    time_stepping: str,
    min_width: float,
) -> tuple[list[float], list[int]]:
    """Compute stepping data (cached on parameters).

    Parameters
    ----------
    ts : tuple of float
        Snapshot scale factors from the integration settings (e.g. ``(1.0,)``).
        Passed directly to ``simulation_stepping`` so the step count is driven by
        *nb_steps*, not by a ``nb_shells`` lightcone geometry.

    Returns
    -------
    steps : list[float]
        All visited scale factors (including t0 and every snapshot target).
    snapshot_indices : list[int]
        Positions inside *steps* that correspond to a snapshot target.
    """
    jc, simulation_stepping = _load_jax_modules()
    import jax.numpy as jnp

    cosmo = jc.Planck18()
    ts_arr = jnp.asarray(ts) if ts else None

    result = simulation_stepping(
        cosmo, t0, t1, nb_steps,
        ts=ts_arr,
        time_stepping=time_stepping,
        min_width=min_width,
    )
    steps = jnp.asarray(result).tolist()

    # Identify which steps land exactly on a snapshot target
    snapshot_indices: list[int] = []
    if ts:
        tol = 1e-8
        for snap_a in ts:
            for i, a in enumerate(steps):
                if abs(a - snap_a) < tol:
                    snapshot_indices.append(i)
                    break  # each target appears at most once

    return steps, snapshot_indices


def render_stepping_plot(
    sim_params: dict,
    lightcone_params: dict,
    box_sizes: list[float],
    observer_position: list[float],
    time_stepping: str,
    min_width: float,
) -> None:
    """Render an interactive stepping plot in the current Streamlit container."""
    import plotly.graph_objects as go
    import numpy as np

    st.subheader("Simulation Stepping Plot")

    t0 = sim_params.get("t0", 0.1)
    t1 = sim_params.get("t1", 1.0)
    nb_steps = sim_params.get("nb_steps", 30)

    # Read snapshot ts from integration settings; fall back to [t1] (single snapshot)
    ts_raw = sim_params.get("ts", [t1])
    if ts_raw is None:
        ts_raw = [t1]
    ts: tuple[float, ...] = tuple(float(v) for v in np.atleast_1d(ts_raw))

    box = np.asarray(box_sizes)
    observer_pos = np.asarray(observer_position)
    factors = np.clip(observer_pos, 0.0, 1.0)
    factors = 1.0 + 2.0 * np.minimum(factors, 1.0 - factors)
    max_box_size = float(np.min(box / factors))

    st.caption(
        f"t0={t0}, t1={t1}, steps={nb_steps}, ts={list(ts)}, max_box={max_box_size:.1f}"
    )

    if st.button("Compute Stepping Plot", key="compute_stepping"):
        try:
            jc, _ = _load_jax_modules()
            steps, snapshot_indices = _compute_stepping(
                t0, t1, nb_steps, ts, time_stepping, min_width
            )

            a_values = np.array(steps)
            distances = np.array(
                jc.background.radial_comoving_distance(jc.Planck18(), a_values)
            )
            x_all = list(range(len(distances)))

            # -- Snapshot subset --
            snap_x = [x_all[i] for i in snapshot_indices]
            snap_y = [distances[i] for i in snapshot_indices]

            fig = go.Figure()

            # Blue line + round markers for every integration step
            fig.add_trace(go.Scatter(
                x=x_all,
                y=distances,
                mode="lines+markers",
                marker=dict(size=7, symbol="circle", color="#2563EB"),
                line=dict(color="#2563EB", width=1.5),
                name="Integration steps",
            ))

            # Red round markers for snapshot targets
            fig.add_trace(go.Scatter(
                x=snap_x,
                y=snap_y,
                mode="markers",
                marker=dict(size=11, symbol="circle", color="#DC2626",
                            line=dict(color="white", width=1.5)),
                name="Snapshots (ts)",
            ))

            fig.add_hline(
                y=max_box_size,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Max box = {max_box_size:.0f}",
                annotation_position="top right",
            )
            fig.update_layout(
                xaxis_title="Step index",
                yaxis_title="Comoving distance [Mpc/h]",
                title="Simulation Stepping",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=40, r=20, t=40, b=40),
                height=550,
            )
            # Persist figure so it survives reruns.
            st.session_state[_PLOT_KEY] = fig
        except Exception as e:
            st.error(f"Failed to compute stepping plot: {e}")

    # Display persisted figure (or prompt).
    fig_cached = st.session_state.get(_PLOT_KEY)
    if fig_cached is not None:
        st.plotly_chart(fig_cached, use_container_width=True)
    else:
        st.info("Click **Compute Stepping Plot** to generate the visualisation.")