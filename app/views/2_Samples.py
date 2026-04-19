"""Samples page — `fli-launcher -- fli-samples` (one job per submit)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.integration_form import render_integration_form
from app.components.output_form import render_output_sample_form
from app.components.prior_cosmo_form import render_prior_cosmo_form
from app.components.simulation_settings_form import render_simulation_settings
from app.components.slurm_form import render_slurm_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Samples")

top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Generate mock samples (unconditioned) from the simulation prior.\n\n"
        "One `fli-launcher … -- fli-samples …` job per submit (no sweeps)."
    )

cmd_placeholder = st.empty()

c1, c2, c3 = st.columns([1, 1, 1])

with c2:
    integration = render_integration_form(
        prefix="samp_",
        default_sim_type="lensing",
        default_nb_shells=8,
        default_nb_steps=30,
    )

with c1:
    slurm = render_slurm_form(
        defaults={"pdim": [4, 1], "nodes": 1, "gpus_per_node": 4},
        prefix="samp_",
        show_tasks_per_node=False,
    )
    sim = render_simulation_settings(
        prefix="samp_",
        px=slurm["pdim"][0],
        py=slurm["pdim"][1],
        simulation_type=integration["sim_mode"],
        defaults={
            "mx": 64,
            "my": 64,
            "mz": 64,
            "bx": 250.0,
            "by": 250.0,
            "bz": 250.0,
            "nside": 64,
        },
    )

with c3:
    output = render_output_sample_form(prefix="samp_")
    cosmo = render_prior_cosmo_form(prefix="samp_", show_ic=True)

    with st.container(border=True):
        st.subheader("Samples Settings")
        model = st.selectbox("model", ["full", "mock"], index=1, key="samp_model")
        num_samples = st.number_input(
            "num_samples", min_value=1, value=10, key="samp_num_samples"
        )
        sigma_e = st.number_input(
            "sigma_e", min_value=0.0, value=0.26, format="%.4f", key="samp_sigma_e"
        )
        batch_id = st.number_input(
            "batch_id",
            min_value=0,
            value=0,
            key="samp_batch_id",
            help="Single batch index for this job (no sweeping — script separately).",
        )

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

params = {**slurm}
params.update(
    {
        # Integration
        "sim_mode": integration["sim_mode"],
        "lpt_order": integration["lpt_order"],
        "nb_steps": integration["nb_steps"],
        "t0": integration["t0"],
        "t1": integration["t1"],
        "interp": integration["interp"],
        "solver": integration["solver"],
        "time_stepping": integration["time_stepping"],
        "shell_spacing": integration["shell_spacing"],
        "dealiased": integration["dealiased"],
        "exact_growth": integration["exact_growth"],
        "laplace_fd": integration["laplace_fd"],
        "gradient_order": integration["gradient_order"],
        "nb_shells": integration["nb_shells_for_cmd"],
        "ts": integration["ts"],
        "ts_near": integration["ts_near"],
        "ts_far": integration["ts_far"],
        "drift_on_lightcone": integration["drift_on_lightcone"],
        "min_width": integration["min_width"],
        # Lensing
        **{
            k: v
            for k, v in integration.items()
            if k in ("nz_shear", "min_z", "max_z", "n_integrate")
        },
        # Simulation settings
        "mesh_size": sim["mesh_size"],
        "box_size": sim["box_size"],
        "halo_multiplier": sim["halo_multiplier"],
        "observer_position": sim["observer_position"],
        "scheme": sim["scheme"],
        "paint_nside": sim["paint_nside"],
        "kernel_width_arcmin": sim["kernel_width_arcmin"],
        "enable_x64": sim["enable_x64"],
        "nside": sim.get("nside"),
        "flatsky_npix": sim.get("flatsky_npix"),
        "field_size": sim.get("field_size"),
        "density": sim.get("density", False),
        # Samples-specific
        "path": output["output_dir"],
        "model": model,
        "num_samples": num_samples,
        "sigma_e": sigma_e,
        "batch_id": batch_id,
        # Cosmo / IC prior
        "sample": cosmo.get("sample", ["cosmo", "ic"]),
        "prior_omega_c": cosmo.get("prior_omega_c"),
        "prior_sigma8": cosmo.get("prior_sigma8"),
        "prior_h": cosmo.get("prior_h"),
        "prior_ic_gaussian": cosmo.get("prior_ic_gaussian"),
        "initial_condition": cosmo.get("initial_condition"),
    }
)
cmd = build_command("samples", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
