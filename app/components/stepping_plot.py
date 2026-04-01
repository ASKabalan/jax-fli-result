"""Stepping plot component — visualizes simulation time-stepping geometry."""
from __future__ import annotations

import streamlit as st


@st.cache_resource
def _load_jax_modules():
    """Lazy-load heavy JAX dependencies once."""
    import jax_cosmo as jc
    from jax_fli.pm._resolve_geometry import simulation_stepping
    return jc, simulation_stepping


@st.cache_data
def _compute_stepping(t0: float, t1: float, nb_steps: int, nb_shells: int, max_box_size: float):
    """Compute stepping data (cached on parameters)."""
    jc, simulation_stepping = _load_jax_modules()
    import jax.numpy as jnp

    cosmo = jc.Planck18()
    result = simulation_stepping(
        cosmo, t0, t1, nb_steps,
        nb_shells=nb_shells,
        max_comoving_distance=max_box_size,
        time_stepping="log_a",
    )
    return jnp.asarray(result).tolist()


def render_stepping_plot(sim_params: dict, lightcone_params: dict, box_sizes: list[float]) -> None:
    """Render an interactive stepping plot in the current Streamlit container."""
    import plotly.graph_objects as go

    st.subheader("Simulation Stepping Plot")

    t0 = sim_params.get("t0", 0.1)
    t1 = sim_params.get("t1", 1.0)
    nb_steps = sim_params.get("nb_steps", 30)
    nb_shells = lightcone_params.get("nb_shells", 10)
    max_box_size = max(box_sizes) if box_sizes else 1000.0

    st.caption(f"t0={t0}, t1={t1}, steps={nb_steps}, shells={nb_shells}, max_box={max_box_size}")

    if st.button("Compute Stepping Plot", key="compute_stepping"):
        try:
            import numpy as np

            jc, _ = _load_jax_modules()
            steps = _compute_stepping(t0, t1, nb_steps, nb_shells, max_box_size)

            a_values = np.array(steps)
            distances = jc.background.radial_comoving_distance(jc.Planck18(), a_values)
            distances = np.array(distances)
            step_indices = list(range(len(distances)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=step_indices,
                y=distances,
                mode="lines+markers",
                marker=dict(size=5),
                name="Comoving distance",
            ))
            fig.add_hline(
                y=max_box_size,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max box = {max_box_size}",
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
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to compute stepping plot: {e}")
    else:
        st.info("Click 'Compute Stepping Plot' to generate the visualization.")
