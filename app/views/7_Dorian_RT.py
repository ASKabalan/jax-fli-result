"""Dorian RT page — mirrors `fli-launch dorian-rt`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.lensing_form import render_lensing_form
from app.components.lensing_postproc_form import render_lensing_postproc_form
from app.components.slurm_form import render_slurm_form
from app.components.source_form import render_source_form
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Dorian RT")

slurm = render_slurm_form(
    defaults={
        "constraint": "",
        "cpus_per_node": 24,
        "tasks_per_node": 4,
        "gpus_per_node": 0,
        "nodes": 1,
        "qos": "qos_cpu",
        "time_limit": "01:00:00",
    },
    prefix="dor_",
    show_tasks_per_node=True,
    show_pdim=False,
)
# Ray-tracing integrates through high-z shells → default the n(z) ceiling to 3.0 (vs Born's 1.5).
lensing = render_lensing_form(prefix="dor_", defaults={"max_z": 3.0})
source = render_source_form(prefix="dor_")
postproc = render_lensing_postproc_form(
    prefix="dor_", output_default="results/lensing/multi_shell_raytrace"
)

with st.container(border=True):
    st.subheader("Dorian RT-specific")
    name = (
        st.text_input(
            "name",
            value="",
            key="dor_name",
            help="Label stored as AbstractField.name inside the output catalog.",
        ).strip()
        or None
    )
    rt_interp = st.selectbox(
        "rt_interp", ["bilinear", "ngp", "nufft"], key="dor_rt_interp"
    )
    no_parallel_transport = st.checkbox("no_parallel_transport", key="dor_no_pt")
    with_born = st.checkbox(
        "with_born",
        key="dor_with_born",
        help="Also emit the Born convergence byproduct from the same dorian pass",
    )

# Build command — keys mirror _SUBCOMMAND_SPECS["dorian-rt"] in command_builder.py
params = {**slurm, **lensing, **source, **postproc}
params.update(
    {
        "name": name,
        "rt_interp": rt_interp,
        "no_parallel_transport": no_parallel_transport,
        "with_born": with_born,
    }
)
cmd = build_command("dorian-rt", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
