"""Simulate page — mirrors `python -m launcher simulate`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.dynamic_list import render_dynamic_list, render_dynamic_triple_list
from app.components.slurm_form import render_slurm_form
from app.components.sim_form import render_sim_form
from app.components.lensing_form import render_lensing_form
from app.components.lightcone_form import render_lightcone_form
from app.components.stepping_plot import render_stepping_plot
from app.components.command_builder import build_command

st.set_page_config(page_title="Simulate", layout="wide")
inject_custom_css()
st.title("Simulate")

# ── Top zone: SLURM + Sim (left) │ Stepping plot fills right ────────────────
top_left, top_right = st.columns([2, 3])

with top_left:
    slurm = render_slurm_form(prefix="sim_")
    sim = render_sim_form(prefix="sim_")

# ── Bottom zone: 4 columns ───────────────────────────────────────────────────
bot1, bot2, bot3, bot4 = st.columns([1, 1, 1, 1.5])

with bot1:
    lensing = render_lensing_form(prefix="sim_")

with bot2:
    lightcone = render_lightcone_form(defaults={"drift_on_lightcone": True}, prefix="sim_")

with bot3:
    with st.container(border=True):
        st.subheader("Simulation Settings")
        output_dir = st.text_input("output_dir", value="results/cosmology_runs", key="sim_output_dir")
        simulation_type = st.selectbox("simulation_type", ["lpt", "nbody", "lensing"], index=1, key="sim_simulation_type")
        nside = st.number_input("nside", min_value=1, value=64, key="sim_nside")
        c1, c2 = st.columns(2)
        with c1:
            shell_spacing = st.selectbox("shell_spacing", ["comoving", "equal_vol", "a", "growth"], key="sim_shell_spacing")
        with c2:
            solver = st.selectbox("solver", ["kdk", "dkd", "bf"], key="sim_solver")

with bot4:
    with st.container(border=True):
        st.subheader("Cosmology & Seeds")
        mesh_size = render_dynamic_triple_list(
            "mesh_size", "sim_mesh_size",
            [(64, 64, 64), (32, 32, 32)], cast_fn=int,
        )
        box_size = render_dynamic_triple_list(
            "box_size", "sim_box_size",
            [(1000, 1000, 1000)], cast_fn=float,
        )
        omega_c = render_dynamic_list("omega_c", "sim_omega_c", [0.2589], cast_fn=float)
        sigma8 = render_dynamic_list("sigma8", "sim_sigma8", [0.8159], cast_fn=float)
        seed = render_dynamic_list("seed", "sim_seed", [0], cast_fn=int)

# Plot fills the right column — rendered after bot columns so variables are defined
with top_right:
    if simulation_type in ("nbody", "lensing"):
        render_stepping_plot(sim, lightcone, box_size)

# Build command
params = {**slurm, **sim, **lensing, **lightcone}
params.update({
    "output_dir": output_dir,
    "simulation_type": simulation_type,
    "nside": nside,
    "shell_spacing": shell_spacing,
    "solver": solver,
    "mesh_size": mesh_size,
    "box_size": box_size,
    "omega_c": omega_c,
    "sigma8": sigma8,
    "seed": seed,
})
cmd = build_command("simulate", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
