"""Full Field Inference page — mirrors `fli-launcher infer`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.integration_form import render_integration_form
from app.components.output_form import render_infer_config_form
from app.components.prior_cosmo_form import render_prior_cosmo_form
from app.components.simulation_settings_form import render_simulation_settings
from app.components.slurm_form import render_slurm_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Full Field Inference")

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
c1, c2, c3 = st.columns([1, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Integration Settings
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    integration = render_integration_form(
        prefix="inf_",
        default_sim_type="lensing",
        default_nb_shells=8,
        default_nb_steps=40,
    )

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM + Simulation Settings
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 1, "nodes": 1, "pdim": [1, 1]},
        prefix="inf_slurm_",
        show_tasks_per_node=False,
    )
    sim = render_simulation_settings(
        prefix="inf_sim_",
        px=slurm["pdim"][0],
        py=slurm["pdim"][1],
        simulation_type=integration["sim_mode"],
        defaults={
            "mx": 16,
            "my": 16,
            "mz": 16,
            "bx": 1000.0,
            "by": 1000.0,
            "bz": 1000.0,
            "enable_x64": True,
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# c3 — Inference Config + Cosmology & IC
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    infer_cfg = render_infer_config_form(prefix="inf_")
    cosmo = render_prior_cosmo_form(prefix="inf_", show_ic=True)

# ── Stepping plot ─────────────────────────────────────────────────────────────
with top_right:
    render_stepping_plot(
        {
            "t0": integration["t0"],
            "t1": integration["t1"],
            "nb_steps": integration["nb_steps"],
        },
        {"nb_shells": integration["nb_shells"]},
        sim["box_size"],
        sim["observer_position"],
        time_stepping=integration["time_stepping"],
        min_width=integration["min_width"],
    )

# ── Build command ─────────────────────────────────────────────────────────────
params = {**slurm}
params.update(
    {
        **{
            k: v
            for k, v in integration.items()
            if k in ("nz_shear", "min_z", "max_z", "n_integrate")
        },
        # Integration
        "lpt_order": integration["lpt_order"],
        "nb_steps": integration["nb_steps"],
        "t0": integration["t0"],
        "t1": integration["t1"],
        "interp": integration["interp"],
        "scheme": sim["scheme"],
        "paint_nside": sim["paint_nside"],
        "time_stepping": integration["time_stepping"],
        "dealiased": integration["dealiased"],
        "exact_growth": integration["exact_growth"],
        "laplace_fd": integration["laplace_fd"],
        "gradient_order": integration["gradient_order"],
        "enable_x64": sim["enable_x64"],
        # Lightcone
        "nb_shells": integration["nb_shells_for_cmd"],
        "halo_multiplier": sim["halo_multiplier"],
        "observer_position": sim["observer_position"],
        "ts": integration["ts"],
        "ts_near": integration["ts_near"],
        "ts_far": integration["ts_far"],
        "drift_on_lightcone": integration["drift_on_lightcone"],
        "min_width": integration["min_width"],
        # Infer-specific
        **infer_cfg,
        "mesh_size": sim["mesh_size"],
        "box_size": sim["box_size"],
        "sample": cosmo.get("sample", []),
        "omega_c": cosmo["omega_c"],
        "sigma8": cosmo["sigma8"],
        "h": cosmo["h"],
        "initial_condition": cosmo.get("initial_condition"),
    }
)
cmd = build_command("infer", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
