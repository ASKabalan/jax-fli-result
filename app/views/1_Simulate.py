"""Simulate page — `fli-launcher simulate` (per job)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import DEFAULT_NAME_TEMPLATE, build_command
from app.components.dynamic_list import render_dynamic_list, render_dynamic_triple_list
from app.components.lensing_form import render_lensing_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Simulate")

# ── TOP ROW: description (left) │ stepping plot (right) ──────────────────────
top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Submit N-Body cosmological simulation jobs to the cluster.\n\n"
        "One `fli-launcher simulate` job per parameter combination — supports profiling."
    )

# ── MIDDLE: placeholder for generated command (visually between top and bottom)
cmd_placeholder = st.empty()

# ── BOTTOM: 3 columns ────────────────────────────────────────────────────────
# Fill order is chosen to satisfy Python variable dependencies:
#   c2 → defines simulation_type, nb_shells, nb_steps, t0, t1, etc.
#   c1 → defines dispatch_mode, nodes, gpus, pdim, mesh_size, box_size, seed, enable_x64
#   c3 → defines output target, output_dir, profile, then cosmology
# Streamlit always renders columns left→right visually regardless of fill order.
c1, c2, c3 = st.columns([1, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Integration Settings
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    with st.container(border=True):
        st.subheader("Integration Settings")

        # Simulation type — horizontal radio acting as toggle buttons
        _sim_type_label = st.radio(
            "sim_type",
            ["LPT", "PM", "Lensing"],
            index=1,
            horizontal=True,
            label_visibility="collapsed",
            key="sim_simulation_type_radio",
        )
        simulation_type = {"LPT": "lpt", "PM": "nbody", "Lensing": "lensing"}[
            _sim_type_label
        ]
        nbody_active = simulation_type in ("nbody", "lensing")

        st.divider()

        # Snapshot times — dropdown replaces old nb_shells + checkbox pattern
        snapshot_mode = st.selectbox(
            "Snapshot times",
            ["Number of shells", "Specific times", "Near and Far lists"],
            key="sim_snapshot_mode",
            help="Choose how shell boundaries are specified",
        )

        nb_shells = 10  # fallback default
        nb_steps = 30  # fallback default (redefined below per mode)
        ts = ts_near = ts_far = density_widths = None
        nb_shells_for_cmd = None

        if snapshot_mode == "Number of shells":
            snap_col, step_col = st.columns(2)
            with snap_col:
                nb_shells = st.number_input(
                    "nb_shells",
                    min_value=1,
                    value=10,
                    key="sim_nb_shells",
                    help="Number of lightcone snapshot shells",
                )
            with step_col:
                nb_steps = st.number_input(
                    "Time steps",
                    min_value=1,
                    value=30,
                    disabled=not nbody_active,
                    key="sim_nb_steps",
                    help="nb_steps: NBody timesteps. Not used for LPT.",
                )
            nb_shells_for_cmd = nb_shells

        elif snapshot_mode == "Specific times":
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=30,
                disabled=not nbody_active,
                key="sim_nb_steps",
                help="nb_steps: NBody timesteps. Not used for LPT.",
            )
            ts = render_dynamic_list("ts", "sim_ts", [], cast_fn=float) or None
            st.caption(
                "density_widths: one value for all shells, one per shell, or empty (auto-computed from ts)"
            )
            density_widths = (
                render_dynamic_list(
                    "density_widths",
                    "sim_density_widths",
                    [],
                    cast_fn=float,
                )
                or None
            )

        else:  # Near and Far lists
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=30,
                disabled=not nbody_active,
                key="sim_nb_steps",
                help="nb_steps: NBody timesteps. Not used for LPT.",
            )
            ts_near = (
                render_dynamic_list("ts_near", "sim_ts_near", [], cast_fn=float) or None
            )
            ts_far = (
                render_dynamic_list("ts_far", "sim_ts_far", [], cast_fn=float) or None
            )
            st.caption(
                "density_widths: one value for all shells, one per shell, or empty (auto-computed from ts)"
            )
            density_widths = (
                render_dynamic_list(
                    "density_widths",
                    "sim_density_widths",
                    [],
                    cast_fn=float,
                )
                or None
            )

        st.divider()

        min_width = st.number_input(
            "min_width",
            value=50.0,
            format="%.1f",
            key="sim_min_width",
        )
        drift_on_lightcone = st.checkbox(
            "drift on lightcone",
            value=False,
            key="sim_drift_on_lightcone",
        )
        _SHELL_LABELS = [
            "r (comoving distance)",
            "V (equal volume r³)",
            "a (scale factors)",
            "D (growth)",
        ]
        _SHELL_KEYS = ["comoving", "equal_vol", "a", "growth"]
        _shell_label = st.selectbox(
            "shell_spacing",
            _SHELL_LABELS,
            key="sim_shell_spacing",
        )
        shell_spacing = _SHELL_KEYS[_SHELL_LABELS.index(_shell_label)]

        # Solver — human-readable labels, disabled for LPT
        _SOLVER_LABELS = ["Kick-Drift-Kick", "Drift-Kick-Drift", "BullFrog"]
        _SOLVER_KEYS = ["kdk", "dkd", "bf"]
        _solver_label = st.selectbox(
            "PM solver",
            _SOLVER_LABELS,
            disabled=(simulation_type == "lpt"),
            key="sim_solver",
            help="Disabled for LPT — no N-Body stepping.",
        )
        solver = _SOLVER_KEYS[_SOLVER_LABELS.index(_solver_label)]

        # time_stepping — separate from shell_spacing; BullFrog recommends "D (growth)"
        _TS_LABELS = ["a (scale factors)", "D (growth)", "log_a"]
        _TS_KEYS = ["a", "D", "log_a"]
        _ts_default_index = 1 if solver == "bf" else 0
        _ts_label = st.selectbox(
            "time_stepping",
            _TS_LABELS,
            index=_ts_default_index,
            disabled=(simulation_type == "lpt"),
            key="sim_time_stepping",
            help="Time variable used for integrator stepping. Disabled for LPT.",
        )
        time_stepping = _TS_KEYS[_TS_LABELS.index(_ts_label)]
        if solver == "bf":
            st.info("BullFrog works best with `time_stepping = D (growth)`.")

        st.divider()

        # LPT / NBody simulation parameters
        p1, p2 = st.columns(2)
        with p1:
            lpt_order = st.number_input(
                "LPT order",
                min_value=1,
                max_value=3,
                value=2,
                key="sim_lpt_order",
            )
        with p2:
            t0 = st.number_input(
                "t0",
                min_value=0.001,
                value=0.001,
                format="%.4f",
                key="sim_t0",
            )

        p3, p4 = st.columns(2)
        with p3:
            t1 = st.number_input(
                "t1",
                min_value=0.001,
                value=1.0,
                format="%.4f",
                disabled=not nbody_active,
                key="sim_t1",
            )
        with p4:
            interp = st.selectbox(
                "interp",
                ["none", "onion", "telephoto"],
                disabled=not nbody_active,
                key="sim_interp",
            )

        # LPT subsection
        st.markdown("**LPT**")
        la_col, eg_col = st.columns(2)
        with la_col:
            dealiased = st.checkbox("dealiased", value=False, key="sim_dealiased")
        with eg_col:
            exact_growth = st.checkbox(
                "exact_growth", value=False, key="sim_exact_growth"
            )

        # Forces subsection (applies to solver and LPT)
        st.markdown("**Forces**")
        lf_col, go_col = st.columns(2)
        with lf_col:
            laplace_fd = st.checkbox("laplace_fd", value=False, key="sim_laplace_fd")
        with go_col:
            gradient_order = st.selectbox(
                "gradient_order",
                [1, 0],
                key="sim_gradient_order",
                help="1 = finite-difference, 0 = exact ik",
            )

        # Lensing section — only when lensing type selected
        if simulation_type == "lensing":
            st.divider()
            lensing = render_lensing_form(prefix="sim_")
        else:
            lensing = {}

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM Settings + Simulation Settings (stacked)
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    with st.container(border=True):
        # Title + dispatch mode radio on the same line
        hdr, mode_col = st.columns([1, 2])
        with hdr:
            st.subheader("SLURM")
        with mode_col:
            dispatch_mode = st.radio(
                "Dispatch",
                ["sbatch", "local", "dryrun"],
                index=2,
                horizontal=True,
                label_visibility="collapsed",
                key="sim_dispatch_mode",
            )

        ca, cb = st.columns(2)
        with ca:
            account = st.text_input("Account", value="XXX", key="sim_account")
        with cb:
            constraint = st.text_input("Constraint", value="h100", key="sim_constraint")

        cc, cd = st.columns(2)
        with cc:
            qos = st.text_input("QoS", value="qos_gpu_h100-t3", key="sim_qos")
        with cd:
            time_limit = st.text_input(
                "Time limit", value="01:00:00", key="sim_time_limit"
            )

        # sbatch-only fields
        output_logs = "SLURM_LOGS"
        slurm_script = None
        if dispatch_mode == "sbatch":
            ce, cf = st.columns(2)
            with ce:
                output_logs = st.text_input(
                    "Output logs dir",
                    value="SLURM_LOGS",
                    key="sim_output_logs",
                )
            with cf:
                _raw_script = st.text_input(
                    "SLURM script path",
                    value="",
                    placeholder="$SLURM_SCRIPT (empty = omit)",
                    key="sim_slurm_script",
                    help="Path to a slurm script that help set up the environment and run the command. get inspired by https://gist.github.com/ASKabalan/721209322df82dc1ea2dd2d25af3b7ea#file-slurm-gists",
                )
                slurm_script = _raw_script.strip() or None

        st.markdown("**Compute**")

        cg, ch = st.columns(2)
        with cg:
            gpus_per_node = st.number_input(
                "GPU per node",
                min_value=0,
                value=4,
                key="sim_gpus_per_node",
                help="tasks-per-node = GPUs per node (1 task/GPU is the only supported mode)",
            )
        with ch:
            nodes = st.number_input("Nodes", min_value=1, value=1, key="sim_nodes")

        ci, cj = st.columns(2)
        with ci:
            st.markdown("-")
            cpus_per_node = st.number_input(
                "CPU per node",
                min_value=1,
                value=4,
                key="sim_cpus_per_node",
            )
        with cj:
            st.markdown("**PDIMS**")
            pd1, pd2 = st.columns(2)
            with pd1:
                px = st.number_input("PX", min_value=1, value=4, key="sim_pdim_x")
            with pd2:
                py = st.number_input("PY", min_value=1, value=1, key="sim_pdim_y")

        if px * py != nodes * gpus_per_node:
            st.error(
                f"pdim {px}×{py}={px*py} ≠ nodes×GPUs/node "
                f"{nodes}×{gpus_per_node}={nodes*gpus_per_node}"
            )

    with st.container(border=True):
        st.subheader("Simulation Settings")

        # Mesh sizes + halo multiplier on the same header row
        mh1, mh2 = st.columns([2, 1])
        with mh1:
            st.write("**Mesh sizes**")
        with mh2:
            halo_multiplier = st.number_input(
                "Halo multiplier",
                min_value=0.0,
                value=0.5,
                step=0.05,
                format="%.2f",
                key="sim_halo_multiplier",
                help="Halo size = local_mesh × halo_multiplier",
            )

        mesh_size = render_dynamic_triple_list(
            "mesh_size",
            "sim_mesh_size",
            [(64, 64, 64), (32, 32, 32)],
            cast_fn=int,
        )

        st.write("**Box sizes**")
        box_size = render_dynamic_triple_list(
            "box_size",
            "sim_box_size",
            [(1000, 1000, 1000)],
            cast_fn=float,
        )

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input("OX", value=0.5, format="%.2f", key="sim_obs_x")
        with oc2:
            obs_y = st.number_input("OY", value=0.5, format="%.2f", key="sim_obs_y")
        with oc3:
            obs_z = st.number_input("OZ", value=0.5, format="%.2f", key="sim_obs_z")

        seed = render_dynamic_list("Seed", "sim_seed", ["0"], cast_fn=str)

        st.divider()
        enable_x64 = st.checkbox("enable_x64", value=False, key="sim_enable_x64")

        # Halo multiplier validation: local_mesh × (1 + halo_multiplier) must be integer
        _mesh_triples = [mesh_size[i : i + 3] for i in range(0, len(mesh_size), 3)]
        for _triple in _mesh_triples:
            if len(_triple) < 3:
                continue
            _mx, _my, _ = _triple
            _exp_x = (_mx / px) * (1 + halo_multiplier)
            _exp_y = (_my / py) * (1 + halo_multiplier)
            if abs(_exp_x - round(_exp_x)) > 1e-9 or abs(_exp_y - round(_exp_y)) > 1e-9:
                st.error(
                    f"Mesh {_mx}×{_my}: local_mesh × (1 + halo_multiplier) = "
                    f"{_exp_x:.3f}, {_exp_y:.3f} — must be integer"
                )

# ─────────────────────────────────────────────────────────────────────────────
# c3 — Output Settings (top) + Cosmology (bottom)
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    # ── Output Settings ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Output Settings")

        _out_options = ["Spherical (nside)", "Flat sky", "Density", "Particles"]
        output_target = st.radio(
            "Output target",
            _out_options,
            horizontal=True,
            label_visibility="collapsed",
            key="sim_output_target",
        )
        nside = flatsky_npix = field_size = None
        density_flag = False
        scheme = "bilinear"
        paint_nside = None
        kernel_width_arcmin = None

        if output_target == "Spherical (nside)":
            nside = st.number_input("nside", min_value=1, value=64, key="sim_nside")
            scheme = st.selectbox(
                "scheme",
                ["ngp", "bilinear", "rbf_neighbor"],
                index=1,
                key="sim_scheme",
            )
            paint_nside = st.number_input(
                "paint_nside",
                min_value=1,
                value=64,
                key="sim_paint_nside",
            )
            if scheme == "rbf_neighbor":
                with st.expander("RBF parameters"):
                    st.write("**kernel_width_arcmin**")
                    activate_rbf = st.checkbox(
                        "Activate RBF parameters",
                        value=False,
                        key="sim_activate_rbf",
                    )
                    if activate_rbf:
                        kernel_width_arcmin = st.number_input(
                            "kernel_width_arcmin (arcmin)",
                            value=5.0,
                            min_value=0.01,
                            format="%.2f",
                            key="sim_kernel_width_arcmin",
                        )
        elif output_target == "Flat sky":
            st.write("**Pixels (H × W)**")
            fp1, fp2 = st.columns(2)
            with fp1:
                _fp_h = st.number_input(
                    "H", min_value=1, value=512, key="sim_flatsky_h"
                )
            with fp2:
                _fp_w = st.number_input(
                    "W", min_value=1, value=512, key="sim_flatsky_w"
                )
            flatsky_npix = [_fp_h, _fp_w]

            st.write("**Field size (H × W) deg**")
            ff1, ff2 = st.columns(2)
            with ff1:
                _ff_h = st.number_input("H", min_value=1, value=10, key="sim_field_h")
            with ff2:
                _ff_w = st.number_input("W", min_value=1, value=10, key="sim_field_w")
            field_size = [_ff_h, _ff_w]
            if any(v > 10 for v in field_size):
                st.warning("Flat sky approximation is only reliable up to ~10 degrees.")
        elif output_target == "Density":
            density_flag = True
        # Particles: no extra inputs needed

        output_dir = st.text_input(
            "output_dir",
            value="results/cosmology_runs",
            key="sim_output_dir",
            help=(
                "Placeholders: %constraint%, %mesh_size%, %box_size%, "
                "%nb_steps%, %omega_c%, %sigma8%, %seed%"
            ),
        )

        if "sim_name_template" not in st.session_state:
            st.session_state["sim_name_template"] = DEFAULT_NAME_TEMPLATE
        _nt_col, _nt_btn = st.columns([3, 1])
        with _nt_col:
            name_template = st.text_input(
                "Name template",
                key="sim_name_template",
                help=(
                    "Placeholders: %constraint%, %mesh_size%, %box_size%, "
                    "%nb_steps%, %omega_c%, %sigma8%, %seed%"
                ),
            )
        with _nt_btn:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Default", key="sim_name_template_reset"):
                st.session_state["sim_name_template"] = DEFAULT_NAME_TEMPLATE
                st.rerun()

        # Profile
        _profile_disabled = False
        prof_col, iter_col = st.columns([2, 1])
        with prof_col:
            profile = st.checkbox(
                "Profile",
                value=False,
                disabled=_profile_disabled,
                key="sim_profile",
                help="Enables --perf benchmarking (per-job mode only).",
            )
        with iter_col:
            iterations = None
            if profile and not _profile_disabled:
                iterations = st.number_input(
                    "Iter",
                    min_value=1,
                    value=3,
                    key="sim_iterations",
                )

    # ── Cosmology ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Cosmology")

        # omega_c and sigma8 — prominent, expandable for sweeping
        oc_col, s8_col = st.columns(2)
        with oc_col:
            st.markdown("**omega_c**")
            omega_c = render_dynamic_list(
                "omega_c", "sim_omega_c", ["0.2589"], cast_fn=str
            )
        with s8_col:
            st.markdown("**sigma8**")
            sigma8 = render_dynamic_list(
                "sigma8", "sim_sigma8", ["0.8159"], cast_fn=str
            )

        st.divider()

        # Fixed cosmological parameters
        h1c, h2c = st.columns(2)
        with h1c:
            h = st.number_input("h", value=0.6774, format="%.4f", key="sim_h")
        with h2c:
            omega_b = st.number_input(
                "omega_b", value=0.0486, format="%.4f", key="sim_omega_b"
            )

        wa_col, w0_col = st.columns(2)
        with wa_col:
            wa = st.number_input("wa", value=0.0, format="%.4f", key="sim_wa")
        with w0_col:
            w0 = st.number_input("w0", value=-1.0, format="%.4f", key="sim_w0")

        nu_col, k_col = st.columns(2)
        with nu_col:
            omega_nu = st.number_input(
                "omega_nu", value=0.0, format="%.4f", key="sim_omega_nu"
            )
        with k_col:
            omega_k = st.number_input(
                "omega_k", value=0.0, format="%.4f", key="sim_omega_k"
            )

        n_s = st.number_input("n_s", value=0.9667, format="%.4f", key="sim_n_s")

# ── Stepping plot (all required vars now defined) ─────────────────────────────
_sim_for_plot = {"t0": t0, "t1": t1, "nb_steps": nb_steps}
_lc_for_plot = {"nb_shells": nb_shells}

with top_right:
    if simulation_type in ("nbody", "lensing"):
        observer_position = [obs_x, obs_y, obs_z]
        render_stepping_plot(
            _sim_for_plot,
            _lc_for_plot,
            box_size,
            observer_position,
            time_stepping=time_stepping,
            min_width=min_width,
        )

# ── Build command ─────────────────────────────────────────────────────────────
params = {
    # SLURM
    "mode": dispatch_mode,
    "account": account,
    "constraint": constraint,
    "qos": qos,
    "time_limit": time_limit,
    "output_logs": output_logs,
    "gpus_per_node": gpus_per_node,
    "cpus_per_node": cpus_per_node,
    "tasks_per_node": None,
    "nodes": nodes,
    "slurm_script": slurm_script,
    "pdim": [px, py],
    # Sim
    "lpt_order": lpt_order,
    "nb_steps": nb_steps,
    "t0": t0,
    "t1": t1,
    "interp": interp,
    "scheme": scheme,
    "paint_nside": paint_nside,
    "kernel_width_arcmin": kernel_width_arcmin,
    "enable_x64": enable_x64,
    "dealiased": dealiased,
    "exact_growth": exact_growth,
    "gradient_order": gradient_order,
    "laplace_fd": laplace_fd,
    # Cosmo
    "h": h,
    "omega_b": omega_b,
    "omega_k": omega_k,
    "omega_nu": omega_nu,
    "w0": w0,
    "wa": wa,
    "n_s": n_s,
    # Lensing (empty dict when not lensing mode)
    **lensing,
    # Lightcone
    "nb_shells": nb_shells_for_cmd,
    "halo_multiplier": halo_multiplier,
    "observer_position": [obs_x, obs_y, obs_z],
    "ts": ts,
    "ts_near": ts_near,
    "ts_far": ts_far,
    "drift_on_lightcone": drift_on_lightcone,
    "min_width": min_width,
    # Simulation settings
    "output_dir": output_dir,
    "name_template": name_template,
    "simulation_type": simulation_type,
    "nside": nside,
    "flatsky_npix": flatsky_npix,
    "field_size": field_size,
    "density": density_flag,
    "shell_spacing": shell_spacing,
    "time_stepping": time_stepping,
    "solver": solver,
    "mesh_size": mesh_size,
    "box_size": box_size,
    "omega_c": omega_c,
    "sigma8": sigma8,
    "seed": seed,
    "density_widths": density_widths,
}

params["perf"] = profile
params["iterations"] = iterations if profile else None

cmd = build_command("simulate", params)

# Fill the middle placeholder with the generated command
with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
