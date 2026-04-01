"""Infer page — mirrors `python -m launcher infer`."""
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

st.set_page_config(page_title="Infer", layout="wide")
inject_custom_css()
st.title("Infer")

# ── Top zone: SLURM + Sim (left) │ Stepping plot (right) ────────────────────
top_left, top_right = st.columns([2, 3])

with top_left:
    slurm = render_slurm_form(defaults={"gpus_per_node": 1, "nodes": 1}, prefix="inf_")
    sim = render_sim_form(defaults={"nb_steps": 40, "enable_x64": True}, prefix="inf_")

# ── Bottom zone: 4 columns ───────────────────────────────────────────────────
bot1, bot2, bot3, bot4 = st.columns([1, 1, 1, 1.5])

with bot1:
    lensing = render_lensing_form(prefix="inf_")

with bot2:
    lightcone = render_lightcone_form(prefix="inf_")

with bot3:
    with st.container(border=True):
        st.subheader("Inference Config")
        observable_dir = st.text_input("observable_dir", value="observables", key="inf_observable_dir")
        observable = st.text_input("observable (REQUIRED)", key="inf_observable")
        if not observable:
            st.warning("observable is required")
        output_dir = st.text_input("output_dir", value="results/inference_runs", key="inf_output_dir")
        chain_index = st.number_input("chain_index", min_value=0, value=0, key="inf_chain_index")
        adjoint = st.selectbox("adjoint", ["checkpointed", "recursive"], key="inf_adjoint")
        checkpoints = st.number_input("checkpoints", min_value=1, value=10, key="inf_checkpoints")
        c1, c2 = st.columns(2)
        with c1:
            num_warmup = st.number_input("num_warmup", min_value=0, value=1, key="inf_num_warmup")
        with c2:
            num_samples = st.number_input("num_samples", min_value=1, value=1, key="inf_num_samples")
        batch_count = st.number_input("batch_count", min_value=1, value=2, key="inf_batch_count")
        sampler = st.selectbox("sampler", ["NUTS", "HMC", "MCLMC"], key="inf_sampler")
        backend = st.selectbox("backend", ["numpyro", "blackjax"], index=1, key="inf_backend")
        sigma_e = st.number_input("sigma_e", value=0.26, format="%.4f", key="inf_sigma_e")
        sample = render_dynamic_list("sample", "inf_sample", ["cosmo", "ic"], cast_fn=str)
        use_ic = st.checkbox("Set initial_condition", key="inf_use_ic")
        initial_condition = None
        if use_ic:
            initial_condition = st.text_input("initial_condition path", key="inf_ic_path")
        init_cosmo = st.checkbox("init_cosmo", key="inf_init_cosmo")

with bot4:
    with st.container(border=True):
        st.subheader("Cosmology & Mesh")
        st.write("**mesh_size**")
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

        c3, c4, c5 = st.columns(3)
        with c3:
            omega_c = st.number_input("omega_c", value=0.2589, format="%.4f", key="inf_omega_c")
        with c4:
            sigma8 = st.number_input("sigma8", value=0.8159, format="%.4f", key="inf_sigma8")
        with c5:
            h = st.number_input("h", value=0.6774, format="%.4f", key="inf_h")

        seed = st.number_input("seed", min_value=0, value=0, key="inf_seed")

with top_right:
    render_stepping_plot(sim, lightcone, [bx, by, bz])

# Build command
params = {**slurm, **sim, **lensing, **lightcone}
params.update({
    "observable_dir": observable_dir,
    "observable": observable,
    "output_dir": output_dir,
    "mesh_size": [mx, my, mz],
    "box_size": [bx, by, bz],
    "chain_index": chain_index,
    "adjoint": adjoint,
    "checkpoints": checkpoints,
    "num_warmup": num_warmup,
    "num_samples": num_samples,
    "batch_count": batch_count,
    "sampler": sampler,
    "backend": backend,
    "sigma_e": sigma_e,
    "sample": sample,
    "initial_condition": initial_condition,
    "init_cosmo": init_cosmo,
    "omega_c": omega_c,
    "sigma8": sigma8,
    "h": h,
    "seed": seed,
})
cmd = build_command("infer", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
