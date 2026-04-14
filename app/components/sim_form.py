"""Simulation argument form — mirrors parser.add_common_sim_args."""
from __future__ import annotations

import streamlit as st


def render_sim_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("Simulation")

        c1, c2 = st.columns(2)
        with c1:
            lpt_order = st.number_input(
                "LPT order",
                min_value=1,
                max_value=3,
                value=defaults.get("lpt_order", 2),
                key=f"{prefix}lpt_order",
            )
        with c2:
            nb_steps = st.number_input(
                "Nb steps",
                min_value=1,
                value=defaults.get("nb_steps", 30),
                key=f"{prefix}nb_steps",
            )

        c3, c4 = st.columns(2)
        with c3:
            t0 = st.number_input(
                "t0",
                min_value=0.001,
                value=defaults.get("t0", 0.1),
                format="%.4f",
                key=f"{prefix}t0",
            )
        with c4:
            t1 = st.number_input(
                "t1",
                min_value=0.001,
                value=defaults.get("t1", 1.0),
                format="%.4f",
                key=f"{prefix}t1",
            )

        interp = st.selectbox(
            "Interpolation",
            ["none", "onion", "telephoto"],
            index=["none", "onion", "telephoto"].index(defaults.get("interp", "none")),
            key=f"{prefix}interp",
        )
        scheme = st.selectbox(
            "Scheme",
            ["ngp", "bilinear", "rbf_neighbor"],
            index=["ngp", "bilinear", "rbf_neighbor"].index(
                defaults.get("scheme", "bilinear")
            ),
            key=f"{prefix}scheme",
        )

        use_paint = st.checkbox(
            "Set paint_nside",
            value=defaults.get("paint_nside") is not None,
            key=f"{prefix}use_paint_nside",
        )
        paint_nside = None
        if use_paint:
            paint_nside = st.number_input(
                "paint_nside",
                min_value=1,
                value=defaults.get("paint_nside", 64),
                key=f"{prefix}paint_nside",
            )

        use_kernel = st.checkbox(
            "Set kernel_width_arcmin",
            value=defaults.get("kernel_width_arcmin") is not None,
            key=f"{prefix}use_kernel_width",
        )
        kernel_width_arcmin = None
        if use_kernel:
            kernel_width_arcmin = st.number_input(
                "kernel_width_arcmin (arcmin)",
                min_value=0.0,
                value=defaults.get("kernel_width_arcmin", 1.0),
                format="%.4f",
                key=f"{prefix}kernel_width_arcmin",
            )

        # ── Solver & time stepping ────────────────────────────────────────────
        _SOLVER_LABELS = ["Kick-Drift-Kick", "Drift-Kick-Drift", "BullFrog"]
        _SOLVER_KEYS = ["kdk", "dkd", "bf"]
        _solver_default = defaults.get("solver", "kdk")
        _solver_idx = (
            _SOLVER_KEYS.index(_solver_default)
            if _solver_default in _SOLVER_KEYS
            else 0
        )
        _solver_label = st.selectbox(
            "PM solver", _SOLVER_LABELS, index=_solver_idx, key=f"{prefix}solver"
        )
        solver = _SOLVER_KEYS[_SOLVER_LABELS.index(_solver_label)]

        _TS_LABELS = ["a (scale factors)", "D (growth)", "log_a"]
        _TS_KEYS = ["a", "D", "log_a"]
        _ts_default = defaults.get("time_stepping", "D" if solver == "bf" else "a")
        _ts_idx = _TS_KEYS.index(_ts_default) if _ts_default in _TS_KEYS else 0
        _ts_label = st.selectbox(
            "time_stepping", _TS_LABELS, index=_ts_idx, key=f"{prefix}time_stepping"
        )
        time_stepping = _TS_KEYS[_TS_LABELS.index(_ts_label)]
        if solver == "bf":
            st.info("BullFrog works best with `time_stepping = D (growth)`.")

        # ── LPT & Forces ──────────────────────────────────────────────────────
        st.markdown("**LPT**")
        la_col, eg_col = st.columns(2)
        with la_col:
            dealiased = st.checkbox(
                "dealiased",
                value=defaults.get("dealiased", False),
                key=f"{prefix}dealiased",
            )
        with eg_col:
            exact_growth = st.checkbox(
                "exact_growth",
                value=defaults.get("exact_growth", False),
                key=f"{prefix}exact_growth",
            )

        st.markdown("**Forces**")
        lf_col, go_col = st.columns(2)
        with lf_col:
            laplace_fd = st.checkbox(
                "laplace_fd",
                value=defaults.get("laplace_fd", False),
                key=f"{prefix}laplace_fd",
            )
        with go_col:
            gradient_order = st.selectbox(
                "gradient_order",
                [1, 0],
                index=[1, 0].index(defaults.get("gradient_order", 1)),
                key=f"{prefix}gradient_order",
                help="1 = finite-difference, 0 = exact ik",
            )

        enable_x64 = st.checkbox(
            "Enable x64",
            value=defaults.get("enable_x64", False),
            key=f"{prefix}enable_x64",
        )

        return {
            "lpt_order": lpt_order,
            "nb_steps": nb_steps,
            "t0": t0,
            "t1": t1,
            "interp": interp,
            "scheme": scheme,
            "paint_nside": paint_nside,
            "kernel_width_arcmin": kernel_width_arcmin,
            "solver": solver,
            "time_stepping": time_stepping,
            "dealiased": dealiased,
            "exact_growth": exact_growth,
            "laplace_fd": laplace_fd,
            "gradient_order": gradient_order,
            "enable_x64": enable_x64,
        }
