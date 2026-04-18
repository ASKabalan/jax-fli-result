"""Born RT page — mirrors `fli-launch born-rt`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.lensing_form import render_lensing_form
from app.components.slurm_form import render_slurm_form
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Born RT")

slurm = render_slurm_form(
    defaults={"constraint": "", "pdim": [4, 1], "nodes": 1},
    prefix="born_",
)
lensing = render_lensing_form(prefix="born_")

with st.container(border=True):
    st.subheader("Born RT-specific")
    input_path = st.text_input(
        "input", value="results/cosmology_runs", key="born_input"
    )
    output_path = st.text_input(
        "output", value="results/lensing/multi_shell", key="born_output"
    )
    enable_x64 = st.checkbox("enable_x64", key="born_enable_x64")

# Build command — keys mirror _SUBCOMMAND_SPECS["born-rt"] in command_builder.py
params = {**slurm, **lensing}
params.update(
    {
        "input": input_path,
        "output": output_path,
        "enable_x64": enable_x64,
    }
)
cmd = build_command("born-rt", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
