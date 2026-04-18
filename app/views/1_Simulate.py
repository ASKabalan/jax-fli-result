"""Simulate page — `fli-launcher -- fli-simulate` (one job per submit)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.cosmo_form import render_cosmo_form
from app.components.integration_form import render_integration_form
from app.components.output_form import render_output_form
from app.components.simulation_settings_form import render_simulation_settings
from app.components.slurm_form import render_slurm_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Simulate")

top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Submit one N-Body cosmological simulation job to the cluster.\n\n"
        "`fli-launcher … -- fli-simulate …` — one job per submit (no sweeps)."
    )

cmd_placeholder = st.empty()

c1, c2, c3 = st.columns([1, 1, 1])

with c2:
    integration = render_integration_form(
        prefix="sim_",
        default_sim_type="nbody",
        default_nb_shells=10,
        default_nb_steps=30,
        show_density_widths=True,
    )

with c1:
    slurm = render_slurm_form(
        prefix="sim_",
        show_pdim=True,
        defaults={"nodes": 1, "gpus_per_node": 4, "pdim": [4, 1]},
    )
    sim = render_simulation_settings(
        prefix="sim_",
        px=slurm["pdim"][0],
        py=slurm["pdim"][1],
        defaults={"mx": 64, "my": 64, "mz": 64, "bx": 1000.0, "by": 1000.0, "bz": 1000.0},
    )

with c3:
    output = render_output_form(
        prefix="sim_",
        show_name=True,
        profile=True,
        default_output_dir="results/sim_output.parquet",
    )
    cosmo = render_cosmo_form(prefix="sim_")

with top_right:
    if integration["sim_mode"] in ("pm", "lensing"):
        render_stepping_plot(
            {"t0": integration["t0"], "t1": integration["t1"], "nb_steps": integration["nb_steps"]},
            {"nb_shells": integration["nb_shells"]},
            sim["box_size"],
            sim["observer_position"],
            time_stepping=integration["time_stepping"],
            min_width=integration["min_width"],
        )

params = {
    **slurm,
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
    "gradient_order": integration["gradient_order"],
    "laplace_fd": integration["laplace_fd"],
    "density_widths": integration["density_widths"],
    "ts": integration["ts"],
    "ts_near": integration["ts_near"],
    "ts_far": integration["ts_far"],
    "drift_on_lightcone": integration["drift_on_lightcone"],
    "min_width": integration["min_width"],
    "nb_shells": integration["nb_shells_for_cmd"],
    # Lensing (only populated when sim_mode == "lensing")
    **{k: v for k, v in integration.items() if k in ("nz_shear", "min_z", "max_z", "n_integrate")},
    # Simulation settings
    "mesh_size": sim["mesh_size"],
    "box_size": sim["box_size"],
    "halo_multiplier": sim["halo_multiplier"],
    "observer_position": sim["observer_position"],
    "seed": sim["seed"],
    "scheme": sim["scheme"],
    "paint_nside": sim["paint_nside"],
    "kernel_width_arcmin": sim["kernel_width_arcmin"],
    "enable_x64": sim["enable_x64"],
    "nside": sim.get("nside"),
    "flatsky_npix": sim.get("flatsky_npix"),
    "field_size": sim.get("field_size"),
    "density": sim.get("density", False),
    # Cosmology
    **cosmo,
    # Output
    "output": output["output"],
    "name": output.get("name"),
    "perf": output.get("perf"),
    "iterations": output.get("iterations"),
}

cmd = build_command("simulate", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
