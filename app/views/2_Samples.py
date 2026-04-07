"""Samples page — mirrors `fli-launch samples`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.dynamic_list import render_dynamic_list
from app.components.slurm_form import render_slurm_form
from app.components.sim_form import render_sim_form
from app.components.lensing_form import render_lensing_form
from app.components.lightcone_form import render_lightcone_form
from app.components.stepping_plot import render_stepping_plot
from app.components.command_builder import build_command

inject_custom_css()
st.title("Samples")

# ── Top zone: SLURM + Sim (left) │ Stepping plot (right) ────────────────────
top_left, top_right = st.columns([2, 3])

with top_left:
    slurm = render_slurm_form(prefix="samp_")
    sim = render_sim_form(defaults={"t0": 0.01, "nb_steps": 100}, prefix="samp_")

# ── Bottom zone: 4 columns ───────────────────────────────────────────────────
bot1, bot2, bot3, bot4 = st.columns([1, 1, 1, 1.5])

with bot1:
    lensing = render_lensing_form(prefix="samp_")

with bot2:
    lightcone = render_lightcone_form(defaults={"nb_shells": 8}, prefix="samp_")

with bot3:
    with st.container(border=True):
        st.subheader("Sample Settings")
        output_dir = st.text_input("output_dir", value="test_fli_samples", key="samp_output_dir")
        model = st.selectbox("model", ["full", "mock"], index=1, key="samp_model")
        nside = st.number_input("nside", min_value=1, value=64, key="samp_nside")
        num_samples = st.number_input("num_samples", min_value=1, value=10, key="samp_num_samples")
        chains = render_dynamic_list("chains", "samp_chains", [0, 1, 2, 3], cast_fn=int)
        batches = render_dynamic_list("batches", "samp_batches", [0, 1, 2, 3, 4, 5], cast_fn=int)

with bot4:
    with st.container(border=True):
        st.subheader("Mesh & Box")
        st.write("**mesh_size**")
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

with top_right:
    render_stepping_plot(sim, lightcone, [bx, by, bz])

# Build command
params = {**slurm, **sim, **lensing, **lightcone}
params.update({
    "output_dir": output_dir,
    "model": model,
    "mesh_size": [mx, my, mz],
    "box_size": [bx, by, bz],
    "nside": nside,
    "num_samples": num_samples,
    "chains": chains,
    "batches": batches,
})
cmd = build_command("samples", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
