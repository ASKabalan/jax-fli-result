"""Simulate page — merged `fli-launcher simulate` (per job) and `fli-launcher grid` (single job)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.dynamic_list import render_dynamic_list, render_dynamic_triple_list
from app.components.lensing_form import render_lensing_form
from app.components.stepping_plot import render_stepping_plot
from app.components.command_builder import build_command

st.set_page_config(page_title="Simulate", layout="wide")
inject_custom_css()
st.title("Simulate")

# ── TOP ROW: description + job mode toggle (left) │ stepping plot (right) ───
top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Submit N-Body cosmological simulation jobs to the cluster.\n\n"
        "- **Per job (simulate)**: one `fli-launcher simulate` job per parameter combination — supports profiling.\n"
        "- **Single job (grid)**: one `fli-launcher grid` job that loops over all combinations internally."
    )
    _job_mode_label = st.radio(
        "Grid mode",
        ["Per job (simulate)", "Single job (grid)"],
        horizontal=True,
        label_visibility="collapsed",
        key="sim_job_mode",
    )
    job_mode = "simulate" if _job_mode_label == "Per job (simulate)" else "grid"

# ── MIDDLE: placeholder for generated command (visually between top and bottom)
cmd_placeholder = st.empty()

# ── BOTTOM: 4 equal columns ──────────────────────────────────────────────────
# Fill order is chosen to satisfy Python variable dependencies:
#   c3 → defines simulation_type, nb_shells, nb_steps, t0, t1, etc.
#   c2 → defines box_size, mesh_size, halo_fraction, observer, seed
#   c1 → defines dispatch_mode, nodes, gpus, pdim, output_dir, profile
#   c4 → defines omega_c, sigma8, fixed cosmo params
# Streamlit always renders columns left→right visually regardless of fill order.
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# c3 — Simulation type + lightcone + sim params
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    with st.container(border=True):
        # Simulation type — horizontal radio acting as toggle buttons
        _sim_type_label = st.radio(
            "sim_type",
            ["LPT", "PM", "Lensing"],
            index=1,
            horizontal=True,
            label_visibility="collapsed",
            key="sim_simulation_type_radio",
        )
        simulation_type = {"LPT": "lpt", "PM": "nbody", "Lensing": "lensing"}[_sim_type_label]
        nbody_active = simulation_type in ("nbody", "lensing")

        st.divider()

        # Snapshots (nb_shells) | Time steps (nb_steps) — side by side
        snap_col, step_col = st.columns(2)
        with snap_col:
            nb_shells = st.number_input(
                "Snapshots",
                min_value=1, value=10,
                key="sim_nb_shells",
                help="nb_shells: number of lightcone snapshot shells",
            )
        with step_col:
            nb_steps = st.number_input(
                "Time steps",
                min_value=1, value=30,
                disabled=not nbody_active,
                key="sim_nb_steps",
                help="nb_steps: NBody timesteps. Not used for LPT.",
            )

        # Toggle to use custom ts instead of nb_shells
        use_custom_ts = st.checkbox(
            "Custom ts / ts_near / ts_far instead of nb_shells",
            value=False,
            key="sim_use_custom_ts",
        )
        ts = ts_near = ts_far = None
        if use_custom_ts:
            ts       = render_dynamic_list("ts",      "sim_ts",      [], cast_fn=float) or None
            ts_near  = render_dynamic_list("ts_near", "sim_ts_near", [], cast_fn=float) or None
            ts_far   = render_dynamic_list("ts_far",  "sim_ts_far",  [], cast_fn=float) or None
            nb_shells_for_cmd = None  # ts takes precedence in the launcher
        else:
            nb_shells_for_cmd = nb_shells

        st.divider()

        min_width = st.number_input(
            "min_width", value=50.0, format="%.1f", key="sim_min_width",
        )
        drift_on_lightcone = st.checkbox(
            "drift on lightcone", value=True, key="sim_drift_on_lightcone",
        )
        shell_spacing = st.selectbox(
            "shell_spacing",
            ["comoving", "equal_vol", "a", "growth"],
            key="sim_shell_spacing",
        )
        solver = st.selectbox(
            "PM solver",
            ["kdk", "dkd", "bf"],
            disabled=(simulation_type == "lpt"),
            key="sim_solver",
            help="Disabled for LPT — no N-Body stepping.",
        )

        st.divider()

        # LPT / NBody simulation parameters
        p1, p2 = st.columns(2)
        with p1:
            lpt_order = st.number_input(
                "LPT order", min_value=1, max_value=3, value=2,
                key="sim_lpt_order",
            )
        with p2:
            t0 = st.number_input(
                "t0", min_value=0.001, value=0.1, format="%.4f",
                key="sim_t0",
            )

        p3, p4 = st.columns(2)
        with p3:
            t1 = st.number_input(
                "t1", min_value=0.001, value=1.0, format="%.4f",
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

        scheme = st.selectbox(
            "scheme", ["ngp", "bilinear", "rbf_neighbor"],
            key="sim_scheme",
        )
        enable_x64 = st.checkbox("enable_x64", value=False, key="sim_enable_x64")

        # Lensing section — only when lensing type selected
        if simulation_type == "lensing":
            st.divider()
            lensing = render_lensing_form(prefix="sim_")
        else:
            lensing = {}

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Simulation Settings (mesh, box, observer, seed)
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    with st.container(border=True):
        st.subheader("Simulation Settings")

        # Mesh sizes + halo fraction on the same header row
        mh1, mh2 = st.columns([2, 1])
        with mh1:
            st.write("**Mesh sizes**")
        with mh2:
            halo_fraction = st.number_input(
                "Halo fraction", min_value=1, value=8,
                key="sim_halo_fraction",
            )

        mesh_size = render_dynamic_triple_list(
            "mesh_size", "sim_mesh_size",
            [(64, 64, 64), (32, 32, 32)], cast_fn=int,
        )

        st.write("**Box sizes**")
        box_size = render_dynamic_triple_list(
            "box_size", "sim_box_size",
            [(1000, 1000, 1000)], cast_fn=float,
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

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM Settings
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
            _tl_default = "24:00:00" if job_mode == "grid" else "01:00:00"
            time_limit = st.text_input("Time limit", value=_tl_default, key="sim_time_limit")

        # sbatch-only fields
        output_logs = "SLURM_LOGS"
        slurm_script = None
        if dispatch_mode == "sbatch":
            ce, cf = st.columns(2)
            with ce:
                output_logs = st.text_input(
                    "Output logs dir", value="SLURM_LOGS", key="sim_output_logs",
                )
            with cf:
                _raw_script = st.text_input(
                    "SLURM script path",
                    value="",
                    placeholder="$SLURM_SCRIPT (empty = omit)",
                    key="sim_slurm_script",
                )
                slurm_script = _raw_script.strip() or None

        st.markdown("**Compute**")

        cg, ch = st.columns(2)
        with cg:
            gpus_per_node = st.number_input(
                "GPU per node", min_value=0, value=4,
                key="sim_gpus_per_node",
                help="tasks-per-node = GPUs per node (1 task/GPU is the only supported mode)",
            )
        with ch:
            nodes = st.number_input("Nodes", min_value=1, value=1, key="sim_nodes")

        ci, cj = st.columns(2)
        with ci:
            cpus_per_node = st.number_input(
                "CPU per node", min_value=1, value=4, key="sim_cpus_per_node",
            )
        with cj:
            st.markdown("**PDIMS**")
            pd1, pd2 = st.columns(2)
            with pd1:
                px = st.number_input("PX", min_value=1, value=1, key="sim_pdim_x")
            with pd2:
                py = st.number_input("PY", min_value=1, value=4, key="sim_pdim_y")

        if px * py != nodes * gpus_per_node:
            st.error(
                f"pdim {px}×{py}={px*py} ≠ nodes×GPUs/node "
                f"{nodes}×{gpus_per_node}={nodes*gpus_per_node}"
            )

        st.markdown("**Output**")
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
        if output_target == "Spherical (nside)":
            nside = st.number_input("nside", min_value=1, value=64, key="sim_nside")
        elif output_target == "Flat sky":
            st.write("**Pixels (H × W)**")
            fp1, fp2 = st.columns(2)
            with fp1:
                _fp_h = st.number_input("H", min_value=1, value=512, key="sim_flatsky_h")
            with fp2:
                _fp_w = st.number_input("W", min_value=1, value=512, key="sim_flatsky_w")
            flatsky_npix = [_fp_h, _fp_w]

            st.write("**Field size (H × W) px**")
            ff1, ff2 = st.columns(2)
            with ff1:
                _ff_h = st.number_input("H", min_value=1, value=256, key="sim_field_h")
            with ff2:
                _ff_w = st.number_input("W", min_value=1, value=256, key="sim_field_w")
            field_size = [_ff_h, _ff_w]
        elif output_target == "Density":
            density_flag = True
        # Particles: no extra inputs needed

        _out_dir_default = "results/cosmology_runs" if job_mode == "simulate" else "results/grid_runs"
        output_dir = st.text_input(
            "output_dir", value=_out_dir_default, key="sim_output_dir",
        )

        # Profile (grayed out in grid mode)
        _profile_disabled = (job_mode == "grid")
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
                    "Iter", min_value=1, value=3, key="sim_iterations",
                )

# ─────────────────────────────────────────────────────────────────────────────
# c4 — Cosmology
# ─────────────────────────────────────────────────────────────────────────────
with c4:
    with st.container(border=True):
        st.subheader("Cosmology")

        # omega_c and sigma8 — prominent, expandable for sweeping
        oc_col, s8_col = st.columns(2)
        with oc_col:
            st.markdown("**omega_c**")
            omega_c = render_dynamic_list("omega_c", "sim_omega_c", ["0.2589"], cast_fn=str)
        with s8_col:
            st.markdown("**sigma8**")
            sigma8 = render_dynamic_list("sigma8", "sim_sigma8", ["0.8159"], cast_fn=str)

        st.divider()

        # Fixed cosmological parameters
        h1c, h2c = st.columns(2)
        with h1c:
            h = st.number_input("h", value=0.6774, format="%.4f", key="sim_h")
        with h2c:
            omega_b = st.number_input("omega_b", value=0.0486, format="%.4f", key="sim_omega_b")

        wa_col, w0_col = st.columns(2)
        with wa_col:
            wa = st.number_input("wa", value=0.0, format="%.4f", key="sim_wa")
        with w0_col:
            w0 = st.number_input("w0", value=-1.0, format="%.4f", key="sim_w0")

        nu_col, k_col = st.columns(2)
        with nu_col:
            omega_nu = st.number_input("omega_nu", value=0.0, format="%.4f", key="sim_omega_nu")
        with k_col:
            omega_k = st.number_input("omega_k", value=0.0, format="%.4f", key="sim_omega_k")

        n_s = st.number_input("n_s", value=0.9667, format="%.4f", key="sim_n_s")

        st.divider()
        density_widths = render_dynamic_list(
            "density_widths", "sim_density_widths", [], cast_fn=float,
        ) or None

# ── Stepping plot (all required vars now defined) ─────────────────────────────
_sim_for_plot = {"t0": t0, "t1": t1, "nb_steps": nb_steps}
_lc_for_plot  = {"nb_shells": nb_shells}

with top_right:
    if simulation_type in ("nbody", "lensing"):
        render_stepping_plot(_sim_for_plot, _lc_for_plot, box_size)

# ── Build command ─────────────────────────────────────────────────────────────
params = {
    # SLURM
    "mode":           dispatch_mode,
    "account":        account,
    "constraint":     constraint,
    "qos":            qos,
    "time_limit":     time_limit,
    "output_logs":    output_logs,
    "gpus_per_node":  gpus_per_node,
    "cpus_per_node":  cpus_per_node,
    "tasks_per_node": None,
    "nodes":          nodes,
    "slurm_script":   slurm_script,
    "pdim":           [px, py],
    # Sim
    "lpt_order":      lpt_order,
    "nb_steps":       nb_steps,
    "t0":             t0,
    "t1":             t1,
    "interp":         interp,
    "scheme":         scheme,
    "paint_nside":    None,
    "kernel_width_arcmin": None,
    "enable_x64":     enable_x64,
    # Cosmo
    "h":              h,
    "omega_b":        omega_b,
    "omega_k":        omega_k,
    "omega_nu":       omega_nu,
    "w0":             w0,
    "wa":             wa,
    "n_s":            n_s,
    # Lensing (empty dict when not lensing mode)
    **lensing,
    # Lightcone
    "nb_shells":          nb_shells_for_cmd,
    "halo_fraction":      halo_fraction,
    "observer_position":  [obs_x, obs_y, obs_z],
    "ts":                 ts,
    "ts_near":            ts_near,
    "ts_far":             ts_far,
    "drift_on_lightcone": drift_on_lightcone,
    "min_width":          min_width,
    # Simulation settings
    "output_dir":         output_dir,
    "simulation_type":    simulation_type,
    "nside":              nside,
    "flatsky_npix":       flatsky_npix,
    "field_size":         field_size,
    "density":            density_flag,
    "shell_spacing":      shell_spacing,
    "solver":             solver,
    "mesh_size":          mesh_size,
    "box_size":           box_size,
    "omega_c":            omega_c,
    "sigma8":             sigma8,
    "seed":               seed,
    "density_widths":     density_widths,
}

if job_mode == "simulate":
    params["perf"]       = profile
    params["iterations"] = iterations if profile else None

cmd = build_command(job_mode, params)

# Fill the middle placeholder with the generated command
with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
