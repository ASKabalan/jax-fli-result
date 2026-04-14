"""Infer page — mirrors `fli-launcher infer`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.dynamic_list import render_dynamic_list
from app.components.lensing_form import render_lensing_form
from app.components.slurm_form import render_slurm_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Infer")

# ── TOP ROW: description (left) | stepping plot (right) ──────────────────────
top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Run MCMC inference on a weak-lensing observable.\n\n"
        "Samples cosmological parameters and/or initial conditions using the chosen sampler backend."
    )

# ── MIDDLE: command preview placeholder ──────────────────────────────────────
cmd_placeholder = st.empty()

# ── BOTTOM: 3 columns ─────────────────────────────────────────────────────────
# Fill order: c2 first (defines simulation_type, nb_shells, nb_steps, etc.),
#             then c1 (defines SLURM, mesh/box, observer, scheme),
#             then c3 (inference config + cosmology + IC).
c1, c2, c3 = st.columns([1, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Integration Settings (exact mirror of 1_Simulate)
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    with st.container(border=True):
        st.subheader("Integration Settings")

        _sim_type_label = st.radio(
            "sim_type",
            ["LPT", "PM", "Lensing"],
            index=2,
            horizontal=True,
            label_visibility="collapsed",
            key="inf_simulation_type_radio",
        )
        simulation_type = {"LPT": "lpt", "PM": "nbody", "Lensing": "lensing"}[
            _sim_type_label
        ]
        nbody_active = simulation_type in ("nbody", "lensing")

        st.divider()

        snapshot_mode = st.selectbox(
            "Snapshot times",
            ["Number of shells", "Specific times", "Near and Far lists"],
            key="inf_snapshot_mode",
            help="Choose how shell boundaries are specified",
        )

        nb_shells = 8
        nb_steps = 40
        ts = ts_near = ts_far = None
        nb_shells_for_cmd = None

        if snapshot_mode == "Number of shells":
            snap_col, step_col = st.columns(2)
            with snap_col:
                nb_shells = st.number_input(
                    "nb_shells",
                    min_value=1,
                    value=8,
                    key="inf_nb_shells",
                    help="Number of lightcone snapshot shells",
                )
            with step_col:
                nb_steps = st.number_input(
                    "Time steps",
                    min_value=1,
                    value=40,
                    disabled=not nbody_active,
                    key="inf_nb_steps",
                    help="nb_steps: NBody timesteps. Not used for LPT.",
                )
            nb_shells_for_cmd = nb_shells

        elif snapshot_mode == "Specific times":
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=40,
                disabled=not nbody_active,
                key="inf_nb_steps",
            )
            ts = render_dynamic_list("ts", "inf_ts", [], cast_fn=float) or None

        else:  # Near and Far lists
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=40,
                disabled=not nbody_active,
                key="inf_nb_steps",
            )
            ts_near = (
                render_dynamic_list("ts_near", "inf_ts_near", [], cast_fn=float) or None
            )
            ts_far = (
                render_dynamic_list("ts_far", "inf_ts_far", [], cast_fn=float) or None
            )

        st.divider()

        min_width = st.number_input(
            "min_width",
            value=50.0,
            format="%.1f",
            key="inf_min_width",
        )
        drift_on_lightcone = st.checkbox(
            "drift on lightcone",
            value=False,
            key="inf_drift_on_lightcone",
        )
        equal_vol = st.checkbox(
            "equal_vol",
            value=False,
            key="inf_equal_vol",
            help="Equal-volume shell partitioning",
        )
        _SHELL_LABELS = [
            "r (comoving distance)",
            "V (equal volume r³)",
            "a (scale factors)",
            "D (growth)",
        ]
        _SHELL_KEYS = ["comoving", "equal_vol", "a", "growth"]
        _shell_label = st.selectbox(
            "shell_spacing", _SHELL_LABELS, key="inf_shell_spacing"
        )
        shell_spacing = _SHELL_KEYS[_SHELL_LABELS.index(_shell_label)]

        _SOLVER_LABELS = ["Kick-Drift-Kick", "Drift-Kick-Drift", "BullFrog"]
        _SOLVER_KEYS = ["kdk", "dkd", "bf"]
        _solver_label = st.selectbox(
            "PM solver",
            _SOLVER_LABELS,
            disabled=(simulation_type == "lpt"),
            key="inf_solver",
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
            key="inf_time_stepping",
            help="Time variable used for integrator stepping. Disabled for LPT.",
        )
        time_stepping = _TS_KEYS[_TS_LABELS.index(_ts_label)]
        if solver == "bf":
            st.info("BullFrog works best with `time_stepping = D (growth)`.")

        st.divider()

        p1, p2 = st.columns(2)
        with p1:
            lpt_order = st.number_input(
                "LPT order", min_value=1, max_value=3, value=2, key="inf_lpt_order"
            )
        with p2:
            t0 = st.number_input(
                "t0", min_value=0.001, value=0.01, format="%.4f", key="inf_t0"
            )

        p3, p4 = st.columns(2)
        with p3:
            t1 = st.number_input(
                "t1",
                min_value=0.001,
                value=1.0,
                format="%.4f",
                disabled=not nbody_active,
                key="inf_t1",
            )
        with p4:
            interp = st.selectbox(
                "interp",
                ["none", "onion", "telephoto"],
                disabled=not nbody_active,
                key="inf_interp",
            )

        st.markdown("**LPT**")
        la_col, eg_col = st.columns(2)
        with la_col:
            dealiased = st.checkbox("dealiased", value=False, key="inf_dealiased")
        with eg_col:
            exact_growth = st.checkbox(
                "exact_growth", value=False, key="inf_exact_growth"
            )

        st.markdown("**Forces**")
        lf_col, go_col = st.columns(2)
        with lf_col:
            laplace_fd = st.checkbox("laplace_fd", value=False, key="inf_laplace_fd")
        with go_col:
            gradient_order = st.selectbox(
                "gradient_order",
                [1, 0],
                key="inf_gradient_order",
                help="1 = finite-difference, 0 = exact ik",
            )

        if simulation_type == "lensing":
            st.divider()
            lensing = render_lensing_form(prefix="inf_")
        else:
            lensing = {}

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM + Simulation Settings
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 1, "nodes": 1, "pdim": [1, 1]},
        prefix="inf_slurm_",
        show_tasks_per_node=False,
    )

    with st.container(border=True):
        st.subheader("Simulation Settings")

        mh1, mh2 = st.columns([2, 1])
        with mh1:
            st.write("**mesh_size**")
        with mh2:
            halo_multiplier = st.number_input(
                "Halo mult.",
                min_value=0.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                key="inf_halo_multiplier",
                help="Halo size = local_mesh × halo_multiplier",
            )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            mx = st.number_input("MX", value=16, key="inf_mx")
        with mc2:
            my = st.number_input("MY", value=16, key="inf_my")
        with mc3:
            mz = st.number_input("MZ", value=16, key="inf_mz")

        st.write("**box_size**")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bx = st.number_input("BX", value=1000.0, key="inf_bx")
        with bc2:
            by = st.number_input("BY", value=1000.0, key="inf_by")
        with bc3:
            bz = st.number_input("BZ", value=1000.0, key="inf_bz")

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input("OX", value=0.5, format="%.2f", key="inf_obs_x")
        with oc2:
            obs_y = st.number_input("OY", value=0.5, format="%.2f", key="inf_obs_y")
        with oc3:
            obs_z = st.number_input("OZ", value=0.5, format="%.2f", key="inf_obs_z")

        st.divider()
        scheme = st.selectbox(
            "scheme", ["ngp", "bilinear", "rbf_neighbor"], index=1, key="inf_scheme"
        )
        use_paint = st.checkbox(
            "Set paint_nside", value=False, key="inf_use_paint_nside"
        )
        paint_nside = None
        if use_paint:
            paint_nside = st.number_input(
                "paint_nside", min_value=1, value=64, key="inf_paint_nside"
            )
        enable_x64 = st.checkbox("enable_x64", value=True, key="inf_enable_x64")

# ─────────────────────────────────────────────────────────────────────────────
# c3 — Inference Config + Cosmology & IC
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    with st.container(border=True):
        st.subheader("Inference Config")

        observable_path = st.text_input(
            "observable path",
            value="observables/BORN_SMALL.parquet",
            key="inf_obs_path",
            help="Full path to the observable parquet file",
        )
        _obs_p = (
            Path(observable_path)
            if observable_path
            else Path("observables/BORN_SMALL.parquet")
        )
        observable_dir = str(_obs_p.parent)
        observable = _obs_p.name

        output_dir = st.text_input(
            "output_dir", value="results/inference_runs", key="inf_output_dir"
        )

        ci_col, seed_col = st.columns(2)
        with ci_col:
            chain_index = st.number_input(
                "chain index", min_value=0, value=0, key="inf_chain_index"
            )
        with seed_col:
            seed = st.number_input("seed", min_value=0, value=0, key="inf_seed")

        adjoint = st.selectbox(
            "adjoint", ["checkpointed", "recursive"], key="inf_adjoint"
        )
        checkpoints = st.number_input(
            "checkpoints", min_value=1, value=10, key="inf_checkpoints"
        )

        wm_col, ns_col = st.columns(2)
        with wm_col:
            num_warmup = st.number_input(
                "num_warmup", min_value=0, value=1, key="inf_num_warmup"
            )
        with ns_col:
            num_samples_per_chain = st.number_input(
                "num_samples", min_value=1, value=1, key="inf_num_samples"
            )

        batch_count = st.number_input(
            "batch_count", min_value=1, value=2, key="inf_batch_count"
        )

        sm_col, be_col = st.columns(2)
        with sm_col:
            sampler = st.selectbox(
                "sampler", ["NUTS", "HMC", "MCLMC"], key="inf_sampler"
            )
        with be_col:
            backend = st.selectbox(
                "backend", ["numpyro", "blackjax"], index=1, key="inf_backend"
            )

        sigma_e = st.number_input(
            "sigma_e", value=0.26, format="%.4f", key="inf_sigma_e"
        )
        init_cosmo = st.checkbox(
            "init_cosmo",
            value=False,
            key="inf_init_cosmo",
            help="Warm-start cosmology from observable",
        )

    with st.container(border=True):
        st.subheader("Cosmology")
        st.caption("Uncheck **Fixed** to sample a parameter (sampled by default).")

        def _cosmo_row(label, key, default_val, default_min, default_max):
            val_col, fix_col = st.columns([2, 1])
            with val_col:
                val = st.number_input(
                    label, value=default_val, format="%.4f", key=f"inf_{key}_val"
                )
            with fix_col:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                # sampled by default → Fixed unchecked by default
                fixed = st.checkbox("Fixed", value=False, key=f"inf_{key}_fixed")
            prior_min = prior_max = None
            if not fixed:
                pm_col, px_col = st.columns(2)
                with pm_col:
                    prior_min = st.number_input(
                        "prior min",
                        value=default_min,
                        format="%.4f",
                        key=f"inf_{key}_prior_min",
                    )
                with px_col:
                    prior_max = st.number_input(
                        "prior max",
                        value=default_max,
                        format="%.4f",
                        key=f"inf_{key}_prior_max",
                    )
            return val, fixed, prior_min, prior_max

        omega_c, omega_c_fixed, _, _ = _cosmo_row(
            "omega_c", "omega_c", 0.2589, 0.1, 0.5
        )
        sigma8, sigma8_fixed, _, _ = _cosmo_row("sigma8", "sigma8", 0.8159, 0.4, 1.2)
        h_val, h_fixed, _, _ = _cosmo_row("h", "h", 0.6774, 0.5, 0.9)

        st.divider()
        st.subheader("Initial Conditions")

        # IC is sampled by default (consistent with --sample cosmo ic default)
        ic_fixed = st.checkbox("Fixed (not sampled)", value=False, key="inf_ic_fixed")
        initial_condition = None
        if ic_fixed:
            with st.expander("Truth initial conditions"):
                ic_path = st.text_input("IC parquet path", key="inf_ic_path")
                initial_condition = ic_path if ic_path.strip() else None

# ── Build sample list ─────────────────────────────────────────────────────────
sample = []
if not omega_c_fixed or not sigma8_fixed or not h_fixed:
    sample.append("cosmo")
if not ic_fixed:
    sample.append("ic")

# ── Stepping plot ─────────────────────────────────────────────────────────────
with top_right:
    render_stepping_plot(
        {"t0": t0, "t1": t1, "nb_steps": nb_steps},
        {"nb_shells": nb_shells},
        [bx, by, bz],
        [obs_x, obs_y, obs_z],
        time_stepping=time_stepping,
        min_width=min_width,
    )

# ── Build command ─────────────────────────────────────────────────────────────
params = {**slurm, **lensing}
params.update(
    {
        # Integration
        "lpt_order": lpt_order,
        "nb_steps": nb_steps,
        "t0": t0,
        "t1": t1,
        "interp": interp,
        "scheme": scheme,
        "paint_nside": paint_nside,
        "time_stepping": time_stepping,
        "dealiased": dealiased,
        "exact_growth": exact_growth,
        "laplace_fd": laplace_fd,
        "gradient_order": gradient_order,
        "enable_x64": enable_x64,
        # Lightcone
        "nb_shells": nb_shells_for_cmd,
        "halo_multiplier": halo_multiplier,
        "observer_position": [obs_x, obs_y, obs_z],
        "ts": ts,
        "ts_near": ts_near,
        "ts_far": ts_far,
        "drift_on_lightcone": drift_on_lightcone,
        "equal_vol": equal_vol,
        "min_width": min_width,
        # Infer-specific
        "observable_dir": observable_dir,
        "observable": observable,
        "output_dir": output_dir,
        "mesh_size": [mx, my, mz],
        "box_size": [bx, by, bz],
        "chain_index": chain_index,
        "adjoint": adjoint,
        "checkpoints": checkpoints,
        "num_warmup": num_warmup,
        "num_samples": num_samples_per_chain,
        "batch_count": batch_count,
        "sampler": sampler,
        "backend": backend,
        "sigma_e": sigma_e,
        "sample": sample,
        "initial_condition": initial_condition,
        "init_cosmo": init_cosmo,
        "omega_c": omega_c,
        "sigma8": sigma8,
        "h": h_val,
        "seed": seed,
    }
)
cmd = build_command("infer", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
