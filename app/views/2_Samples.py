"""Samples page — mirrors `fli-launcher samples`."""
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
st.title("Samples")


def _parse_int_range(text: str) -> list[int]:
    """Parse '0,1,2,3' or '0-5' (or mixed '0-3,6,8-9') into a flat int list."""
    result = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            result.append(int(part))
    return result


# ── TOP ROW: description (left) | stepping plot (right) ──────────────────────
top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Generate mock samples (unconditioned) from the simulation prior.\n\n"
        "Jobs are swept over all combinations of **chains × batches**."
    )

# ── MIDDLE: command preview placeholder ──────────────────────────────────────
cmd_placeholder = st.empty()

# ── BOTTOM: 3 columns ─────────────────────────────────────────────────────────
# Fill order: c2 first (defines simulation_type, nb_shells, nb_steps, etc.),
#             then c1 (defines SLURM, mesh/box, observer, scheme),
#             then c3 (output + samples settings).
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
            key="samp_simulation_type_radio",
        )
        simulation_type = {"LPT": "lpt", "PM": "nbody", "Lensing": "lensing"}[
            _sim_type_label
        ]
        nbody_active = simulation_type in ("nbody", "lensing")

        st.divider()

        snapshot_mode = st.selectbox(
            "Snapshot times",
            ["Number of shells", "Specific times", "Near and Far lists"],
            key="samp_snapshot_mode",
            help="Choose how shell boundaries are specified",
        )

        nb_shells = 8
        nb_steps = 100
        ts = ts_near = ts_far = None
        nb_shells_for_cmd = None

        if snapshot_mode == "Number of shells":
            snap_col, step_col = st.columns(2)
            with snap_col:
                nb_shells = st.number_input(
                    "nb_shells",
                    min_value=1,
                    value=8,
                    key="samp_nb_shells",
                    help="Number of lightcone snapshot shells",
                )
            with step_col:
                nb_steps = st.number_input(
                    "Time steps",
                    min_value=1,
                    value=100,
                    disabled=not nbody_active,
                    key="samp_nb_steps",
                    help="nb_steps: NBody timesteps. Not used for LPT.",
                )
            nb_shells_for_cmd = nb_shells

        elif snapshot_mode == "Specific times":
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=100,
                disabled=not nbody_active,
                key="samp_nb_steps",
                help="nb_steps: NBody timesteps. Not used for LPT.",
            )
            ts = render_dynamic_list("ts", "samp_ts", [], cast_fn=float) or None

        else:  # Near and Far lists
            nb_steps = st.number_input(
                "Time steps",
                min_value=1,
                value=100,
                disabled=not nbody_active,
                key="samp_nb_steps",
                help="nb_steps: NBody timesteps. Not used for LPT.",
            )
            ts_near = (
                render_dynamic_list("ts_near", "samp_ts_near", [], cast_fn=float)
                or None
            )
            ts_far = (
                render_dynamic_list("ts_far", "samp_ts_far", [], cast_fn=float) or None
            )

        st.divider()

        min_width = st.number_input(
            "min_width",
            value=50.0,
            format="%.1f",
            key="samp_min_width",
        )
        drift_on_lightcone = st.checkbox(
            "drift on lightcone",
            value=False,
            key="samp_drift_on_lightcone",
        )
        equal_vol = st.checkbox(
            "equal_vol",
            value=False,
            key="samp_equal_vol",
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
            "shell_spacing", _SHELL_LABELS, key="samp_shell_spacing"
        )
        shell_spacing = _SHELL_KEYS[_SHELL_LABELS.index(_shell_label)]

        _SOLVER_LABELS = ["Kick-Drift-Kick", "Drift-Kick-Drift", "BullFrog"]
        _SOLVER_KEYS = ["kdk", "dkd", "bf"]
        _solver_label = st.selectbox(
            "PM solver",
            _SOLVER_LABELS,
            disabled=(simulation_type == "lpt"),
            key="samp_solver",
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
            key="samp_time_stepping",
            help="Time variable used for integrator stepping. Disabled for LPT.",
        )
        time_stepping = _TS_KEYS[_TS_LABELS.index(_ts_label)]
        if solver == "bf":
            st.info("BullFrog works best with `time_stepping = D (growth)`.")

        st.divider()

        p1, p2 = st.columns(2)
        with p1:
            lpt_order = st.number_input(
                "LPT order", min_value=1, max_value=3, value=2, key="samp_lpt_order"
            )
        with p2:
            t0 = st.number_input(
                "t0", min_value=0.001, value=0.01, format="%.4f", key="samp_t0"
            )

        p3, p4 = st.columns(2)
        with p3:
            t1 = st.number_input(
                "t1",
                min_value=0.001,
                value=1.0,
                format="%.4f",
                disabled=not nbody_active,
                key="samp_t1",
            )
        with p4:
            interp = st.selectbox(
                "interp",
                ["none", "onion", "telephoto"],
                disabled=not nbody_active,
                key="samp_interp",
            )

        st.markdown("**LPT**")
        la_col, eg_col = st.columns(2)
        with la_col:
            dealiased = st.checkbox("dealiased", value=False, key="samp_dealiased")
        with eg_col:
            exact_growth = st.checkbox(
                "exact_growth", value=False, key="samp_exact_growth"
            )

        st.markdown("**Forces**")
        lf_col, go_col = st.columns(2)
        with lf_col:
            laplace_fd = st.checkbox("laplace_fd", value=False, key="samp_laplace_fd")
        with go_col:
            gradient_order = st.selectbox(
                "gradient_order",
                [1, 0],
                key="samp_gradient_order",
                help="1 = finite-difference, 0 = exact ik",
            )

        if simulation_type == "lensing":
            st.divider()
            lensing = render_lensing_form(prefix="samp_")
        else:
            lensing = {}

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM + Simulation Settings
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"pdim": [4, 1], "nodes": 1, "gpus_per_node": 4},
        prefix="samp_",
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
                key="samp_halo_multiplier",
                help="Halo size = local_mesh × halo_multiplier",
            )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            mx = st.number_input("MX", value=64, key="samp_mx")
        with mc2:
            my = st.number_input("MY", value=64, key="samp_my")
        with mc3:
            mz = st.number_input("MZ", value=64, key="samp_mz")

        st.write("**box_size**")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bx = st.number_input("BX", value=250.0, key="samp_bx")
        with bc2:
            by = st.number_input("BY", value=250.0, key="samp_by")
        with bc3:
            bz = st.number_input("BZ", value=250.0, key="samp_bz")

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input("OX", value=0.5, format="%.2f", key="samp_obs_x")
        with oc2:
            obs_y = st.number_input("OY", value=0.5, format="%.2f", key="samp_obs_y")
        with oc3:
            obs_z = st.number_input("OZ", value=0.5, format="%.2f", key="samp_obs_z")

        st.divider()
        scheme = st.selectbox(
            "scheme", ["ngp", "bilinear", "rbf_neighbor"], index=1, key="samp_scheme"
        )
        use_paint = st.checkbox(
            "Set paint_nside", value=False, key="samp_use_paint_nside"
        )
        paint_nside = None
        if use_paint:
            paint_nside = st.number_input(
                "paint_nside", min_value=1, value=64, key="samp_paint_nside"
            )
        enable_x64 = st.checkbox("enable_x64", value=False, key="samp_enable_x64")

# ─────────────────────────────────────────────────────────────────────────────
# c3 — Output Settings + Samples Settings
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    with st.container(border=True):
        st.subheader("Output Settings")
        nside = st.number_input("nside", min_value=1, value=64, key="samp_nside")
        output_dir = st.text_input(
            "output_dir", value="test_fli_samples", key="samp_output_dir"
        )

    with st.container(border=True):
        st.subheader("Samples Settings")
        model = st.selectbox("model", ["full", "mock"], index=1, key="samp_model")
        num_samples = st.number_input(
            "num_samples", min_value=1, value=10, key="samp_num_samples"
        )

        st.markdown("**chains** — comma-separated or range (e.g. `0,1,2,3` or `0-3`)")
        chains_text = st.text_input(
            "chains",
            value="0,1,2,3",
            key="samp_chains_text",
            label_visibility="collapsed",
        )
        try:
            chains = _parse_int_range(chains_text)
        except (ValueError, IndexError):
            st.error("Invalid chains format. Use e.g. `0,1,2,3` or `0-3`.")
            chains = []

        st.markdown(
            "**batches** — comma-separated or range (e.g. `0-5` or `0,1,2,3,4,5`)"
        )
        batches_text = st.text_input(
            "batches",
            value="0-5",
            key="samp_batches_text",
            label_visibility="collapsed",
        )
        try:
            batches = _parse_int_range(batches_text)
        except (ValueError, IndexError):
            st.error("Invalid batches format. Use e.g. `0-5` or `0,1,2,3,4,5`.")
            batches = []

# ── Stepping plot (all required vars now defined) ─────────────────────────────
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
        # Samples-specific
        "output_dir": output_dir,
        "model": model,
        "mesh_size": [mx, my, mz],
        "box_size": [bx, by, bz],
        "nside": nside,
        "num_samples": num_samples,
        "chains": chains,
        "batches": batches,
    }
)
cmd = build_command("samples", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
