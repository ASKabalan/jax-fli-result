"""Dorian RT page — mirrors `fli-launch dorian-rt`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.slurm_form import render_slurm_form
from app.components.lensing_form import render_lensing_form
from app.components.command_builder import build_command

st.set_page_config(page_title="Dorian RT", layout="wide")
inject_custom_css()
st.title("Dorian RT")

slurm = render_slurm_form(
    defaults={
        "constraint": "cpu",
        "cpus_per_node": 24,
        "tasks_per_node": 4,
        "nodes": 1,
        "qos": "qos_cpu",
        "time_limit": "01:00:00",
    },
    prefix="dor_",
)
lensing = render_lensing_form(prefix="dor_")

with st.container(border=True):
    st.subheader("Dorian RT-specific")
    input_dir = st.text_input("input_dir", value="results/cosmology_runs", key="dor_input_dir")
    output_dir = st.text_input("output_dir", value="results/lensing/multi_shell_raytrace", key="dor_output_dir")
    rt_interp = st.selectbox("rt_interp", ["bilinear", "ngp", "nufft"], key="dor_rt_interp")
    no_parallel_transport = st.checkbox("no_parallel_transport", key="dor_no_pt")

# Build command
params = {**slurm, **lensing}
params.update({
    "input_dir": input_dir,
    "output_dir": output_dir,
    "rt_interp": rt_interp,
    "no_parallel_transport": no_parallel_transport,
})
cmd = build_command("dorian-rt", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
