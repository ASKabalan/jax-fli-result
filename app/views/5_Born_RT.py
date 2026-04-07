"""Born RT page — mirrors `fli-launch born-rt`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.slurm_form import render_slurm_form
from app.components.lensing_form import render_lensing_form
from app.components.command_builder import build_command

inject_custom_css()
st.title("Born RT")

slurm = render_slurm_form(
    defaults={"constraint": "", "pdim": [4, 1], "nodes": 1},
    prefix="born_",
)
lensing = render_lensing_form(prefix="born_")

with st.container(border=True):
    st.subheader("Born RT-specific")
    input_dir = st.text_input("input_dir", value="results/cosmology_runs", key="born_input_dir")
    output_dir = st.text_input("output_dir", value="results/lensing/multi_shell", key="born_output_dir")
    enable_x64 = st.checkbox("enable_x64", key="born_enable_x64")

# Build command
params = {**slurm, **lensing}
params.update({
    "input_dir": input_dir,
    "output_dir": output_dir,
    "enable_x64": enable_x64,
})
cmd = build_command("born-rt", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
