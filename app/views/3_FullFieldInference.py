"""Full Field Inference page — mirrors `fli-launcher infer`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.forward_model_form import render_forward_model_form
from app.components.integration_form import render_integration_form
from app.components.output_form import render_infer_config_form
from app.components.prior_cosmo_form import render_prior_cosmo_form
from app.components.simulation_settings_form import render_simulation_settings
from app.components.slurm_form import render_slurm_form
from app.components.source_form import render_source_form
from app.components.stepping_plot import render_stepping_plot
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Full Field Inference")

# ── TOP ROW: description (left) | stepping plot (right) ──────────────────────
top_left, top_right = st.columns([1, 1])

with top_left:
    st.markdown(
        "Run MCMC inference on a weak-lensing observable.\n\n"
        "Samples cosmological parameters and/or initial conditions using the chosen sampler."
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
    obs_source = render_source_form(prefix="inf_obs_", title="Observable source")
    ic_source = render_source_form(
        prefix="inf_ic_", title="Initial condition (optional)"
    )
    infer_cfg = render_infer_config_form(prefix="inf_")
    forward_model = render_forward_model_form(prefix="inf_")
    # The IC parquet path is supplied by the dedicated --ic-* source above, so suppress the legacy
    # IC-path field in the prior form (show_ic_path=False).
    cosmo = render_prior_cosmo_form(prefix="inf_", show_ic=True, show_ic_path=False)

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
        # Integration
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
        "paint_order": integration["paint_order"],
        "deconvolution": integration["deconvolution"],
        "nb_shells": integration["nb_shells_for_cmd"],
        "ts": integration["ts"],
        "ts_near": integration["ts_near"],
        "ts_far": integration["ts_far"],
        "drift_on_lightcone": integration["drift_on_lightcone"],
        "min_width": integration["min_width"],
        # Lensing source distribution (--lensing-output / --apodization-scale-deg come from the
        # forward-model form below, not from the lensing / simulation-settings forms).
        **{
            k: v
            for k, v in integration.items()
            if k in ("nz_shear", "min_z", "max_z", "n_integrate")
        },
        # Simulation settings + painting (nside/flatsky come from the observable)
        "mesh_size": sim["mesh_size"],
        "box_size": sim["box_size"],
        "halo_multiplier": sim["halo_multiplier"],
        "observer_position": sim["observer_position"],
        "scheme": sim["scheme"],
        "paint_nside": sim["paint_nside"],
        "kernel_width_arcmin": sim.get("kernel_width_arcmin"),
        "kernel_width_pixels": sim.get("kernel_width_pixels"),
        "pixel_window_deconvolution": sim.get("pixel_window_deconvolution", False),
        "enable_x64": sim["enable_x64"],
        # Observable source (--input / --repo / --data-files)
        **obs_source,
        # Optional initial-condition source (--ic-input / --ic-repo / --ic-data-files)
        **{f"ic_{k}": v for k, v in ic_source.items()},
        # Inference config (path, sampler knobs, no_progress_bar)
        **infer_cfg,
        # Forward-model likelihood (lensing_output / mask / sigma_unobserved / log_lightcone /
        # apodization_scale_deg / map2alm_method)
        **forward_model,
        # Cosmo / IC priors
        "sample": cosmo.get("sample", ["cosmo", "ic"]),
        "prior_omega_c": cosmo.get("prior_omega_c"),
        "prior_sigma8": cosmo.get("prior_sigma8"),
        "prior_h": cosmo.get("prior_h"),
        "prior_ic_gaussian": cosmo.get("prior_ic_gaussian"),
    }
)
cmd = build_command("infer", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
