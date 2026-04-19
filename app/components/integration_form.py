"""Integration settings form — shared between Simulate, Samples, and Infer pages."""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_list
from app.components.lensing_form import render_lensing_form


def render_integration_form(
    prefix: str = "",
    defaults: dict | None = None,
    default_sim_type: str = "lensing",
    default_nb_shells: int = 8,
    default_nb_steps: int = 30,
    show_density_widths: bool = False,
) -> dict:
    """Render the integration settings form.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.
    default_sim_type:
        Initial simulation type: "lpt", "nbody", or "lensing".
    default_nb_shells:
        Default number of shells.
    default_nb_steps:
        Default number of time steps.
    show_density_widths:
        When True, show density_widths input in "Specific times" and
        "Near and Far lists" modes (used by the Simulate page).

    Returns
    -------
    dict with keys:
        simulation_type, nb_shells, nb_steps, ts, ts_near, ts_far,
        density_widths, nb_shells_for_cmd, min_width, drift_on_lightcone,
        shell_spacing, solver, time_stepping, lpt_order, t0, t1, interp,
        dealiased, exact_growth, laplace_fd, gradient_order.
        shell_spacing replaces the deprecated equal_vol flag.
        + lensing keys (nz_shear, min_z, max_z, n_integrate) when lensing.
    """
    defaults = defaults or {}
    _sim_type_map = {"LPT": "lpt", "PM": "pm", "Lensing": "lensing"}
    _sim_type_labels = list(_sim_type_map.keys())
    _default_sim_label = {v: k for k, v in _sim_type_map.items()}.get(
        defaults.get("simulation_type", default_sim_type), "Lensing"
    )

    with st.container(border=True):
        st.subheader("Integration Settings")

        _sim_type_label = st.radio(
            "sim_type",
            _sim_type_labels,
            index=_sim_type_labels.index(_default_sim_label),
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}simulation_type_radio",
        )
        simulation_type = _sim_type_map[_sim_type_label]
        nbody_active = simulation_type in ("pm", "lensing")

        st.divider()

        snapshot_mode = st.selectbox(
            "Snapshot times",
            ["Number of shells", "Specific times", "Near and Far lists"],
            key=f"{prefix}snapshot_mode",
            help="Choose how shell boundaries are specified",
        )

        nb_shells = defaults.get("nb_shells", default_nb_shells)
        nb_steps = defaults.get("nb_steps", default_nb_steps)
        ts = ts_near = ts_far = density_widths = None
        nb_shells_for_cmd = None

        if snapshot_mode == "Number of shells":
            snap_col, step_col = st.columns(2)
            with snap_col:
                nb_shells = st.number_input(
                    "nb_shells",
                    min_value=1,
                    value=int(defaults.get("nb_shells", default_nb_shells)),
                    key=f"{prefix}nb_shells",
                    help="Number of lightcone snapshot shells",
                )
            with step_col:
                nb_steps = st.number_input(
                    "Time steps",
                    min_value=1,
                    value=int(defaults.get("nb_steps", default_nb_steps)),
                    disabled=not nbody_active,
                    key=f"{prefix}nb_steps",
                    help="NBody timesteps. Not used for LPT.",
                )
            nb_shells_for_cmd = nb_shells

        elif snapshot_mode == "Specific times":
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=int(defaults.get("nb_steps", default_nb_steps)),
                disabled=not nbody_active,
                key=f"{prefix}nb_steps",
                help="NBody timesteps. Not used for LPT.",
            )
            ts = render_dynamic_list("ts", f"{prefix}ts", [], cast_fn=float) or None
            st.caption(
                "density_widths: one value for all shells, one per shell, "
                "or empty (auto-computed from ts)"
            )
            density_widths = (
                render_dynamic_list(
                    "density_widths", f"{prefix}density_widths", [], cast_fn=float
                )
                or None
            )

        else:  # Near and Far lists
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=int(defaults.get("nb_steps", default_nb_steps)),
                disabled=not nbody_active,
                key=f"{prefix}nb_steps",
                help="NBody timesteps. Not used for LPT.",
            )
            ts_near = (
                render_dynamic_list("ts_near", f"{prefix}ts_near", [], cast_fn=float)
                or None
            )
            ts_far = (
                render_dynamic_list("ts_far", f"{prefix}ts_far", [], cast_fn=float)
                or None
            )
            if show_density_widths:
                st.caption(
                    "density_widths: one value for all shells, one per shell, "
                    "or empty (auto-computed from ts)"
                )
                density_widths = (
                    render_dynamic_list(
                        "density_widths", f"{prefix}density_widths", [], cast_fn=float
                    )
                    or None
                )

        st.divider()

        min_width = st.number_input(
            "min_width",
            value=float(defaults.get("min_width", 50.0)),
            format="%.1f",
            key=f"{prefix}min_width",
        )
        drift_on_lightcone = st.checkbox(
            "drift on lightcone",
            value=bool(defaults.get("drift_on_lightcone", False)),
            key=f"{prefix}drift_on_lightcone",
        )

        _SHELL_LABELS = [
            "r (comoving distance)",
            "V (equal volume r³)",
            "a (scale factors)",
            "D (growth)",
        ]
        _SHELL_KEYS = ["comoving", "equal_vol", "a", "growth"]
        _default_shell = defaults.get("shell_spacing", "comoving")
        _shell_idx = (
            _SHELL_KEYS.index(_default_shell) if _default_shell in _SHELL_KEYS else 0
        )
        _shell_label = st.selectbox(
            "shell_spacing",
            _SHELL_LABELS,
            index=_shell_idx,
            key=f"{prefix}shell_spacing",
        )
        shell_spacing = _SHELL_KEYS[_SHELL_LABELS.index(_shell_label)]

        _SOLVER_LABELS = ["Kick-Drift-Kick", "Drift-Kick-Drift", "BullFrog"]
        _SOLVER_KEYS = ["kdk", "dkd", "bf"]
        _default_solver = defaults.get("solver", "kdk")
        _solver_idx = (
            _SOLVER_KEYS.index(_default_solver)
            if _default_solver in _SOLVER_KEYS
            else 0
        )
        _solver_label = st.selectbox(
            "PM solver",
            _SOLVER_LABELS,
            index=_solver_idx,
            disabled=(simulation_type == "lpt"),
            key=f"{prefix}solver",
            help="Disabled for LPT — no N-Body stepping.",
        )
        solver = _SOLVER_KEYS[_SOLVER_LABELS.index(_solver_label)]

        _TS_LABELS = ["a (scale factors)", "D (growth)", "log_a"]
        _TS_KEYS = ["a", "D", "log_a"]
        _ts_default_index = 1 if solver == "bf" else 0
        _ts_label = st.selectbox(
            "time_stepping",
            _TS_LABELS,
            index=_ts_default_index,
            disabled=(simulation_type == "lpt"),
            key=f"{prefix}time_stepping",
            help="Time variable for integrator stepping. Disabled for LPT.",
        )
        time_stepping = _TS_KEYS[_TS_LABELS.index(_ts_label)]
        if solver == "bf":
            st.info("BullFrog works best with `time_stepping = D (growth)`.")

        st.divider()

        p1, p2 = st.columns(2)
        with p1:
            lpt_order = st.number_input(
                "LPT order",
                min_value=1,
                max_value=2,
                value=int(defaults.get("lpt_order", 2)),
                key=f"{prefix}lpt_order",
            )
        with p2:
            t0 = st.number_input(
                "t0",
                min_value=0.001,
                value=float(defaults.get("t0", 0.001)),
                format="%.4f",
                key=f"{prefix}t0",
            )

        p3, p4 = st.columns(2)
        with p3:
            t1 = st.number_input(
                "t1",
                min_value=0.001,
                value=float(defaults.get("t1", 1.0)),
                format="%.4f",
                key=f"{prefix}t1",
            )
        with p4:
            interp = st.selectbox(
                "interp",
                ["none", "onion", "telephoto"],
                key=f"{prefix}interp",
            )

        st.markdown("**LPT**")
        la_col, eg_col = st.columns(2)
        with la_col:
            dealiased = st.checkbox(
                "dealiased",
                value=bool(defaults.get("dealiased", False)),
                key=f"{prefix}dealiased",
            )
        with eg_col:
            exact_growth = st.checkbox(
                "exact_growth",
                value=bool(defaults.get("exact_growth", False)),
                key=f"{prefix}exact_growth",
            )

        st.markdown("**Forces**")
        lf_col, go_col = st.columns(2)
        with lf_col:
            laplace_fd = st.checkbox(
                "laplace_fd",
                value=bool(defaults.get("laplace_fd", False)),
                key=f"{prefix}laplace_fd",
            )
        with go_col:
            gradient_order = st.selectbox(
                "gradient_order",
                [1, 0],
                key=f"{prefix}gradient_order",
                help="1 = finite-difference, 0 = exact ik",
            )

        lensing = {}
        if simulation_type == "lensing":
            st.divider()
            lensing = render_lensing_form(prefix=prefix)

    return {
        "sim_mode": simulation_type,
        "nb_shells": nb_shells,
        "nb_steps": nb_steps,
        "ts": ts,
        "ts_near": ts_near,
        "ts_far": ts_far,
        "density_widths": density_widths,
        "nb_shells_for_cmd": nb_shells_for_cmd,
        "min_width": min_width,
        "drift_on_lightcone": drift_on_lightcone,
        "shell_spacing": shell_spacing,
        "solver": solver,
        "time_stepping": time_stepping,
        "lpt_order": lpt_order,
        "t0": t0,
        "t1": t1,
        "interp": interp,
        "dealiased": dealiased,
        "exact_growth": exact_growth,
        "laplace_fd": laplace_fd,
        "gradient_order": gradient_order,
        **lensing,
    }
