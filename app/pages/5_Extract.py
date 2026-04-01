"""Extract page — mirrors `python -m launcher extract`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css
from app.components.dynamic_list import render_dynamic_list
from app.components.slurm_form import render_slurm_form
from app.components.command_builder import build_command

st.set_page_config(page_title="Extract", layout="wide")
inject_custom_css()
st.title("Extract")

slurm = render_slurm_form(prefix="ext_")

with st.container(border=True):
    st.subheader("Extract-specific")

    source = st.radio("Data source", ["Local directory", "HuggingFace repo"], key="ext_source")

    input_dir = None
    repo_id = None
    config = None

    if source == "Local directory":
        input_dir = st.text_input("input_dir", value="test_fli_samples", key="ext_input_dir")
    else:
        repo_id = st.text_input("repo_id (e.g. 'ASKabalan/jax-fli-experiments')", key="ext_repo_id")
        config = render_dynamic_list(
            "config", "ext_config", [], cast_fn=str,
        )
        config = config or None

    truth_parquet = st.text_input("truth_parquet", value="test_fli_samples/chain_0/samples/samples_0.parquet", key="ext_truth")
    output_file = st.text_input("output_file", value="results/extracts/extract.parquet", key="ext_output")
    set_name = st.text_input("set_name", value="my_extract", key="ext_set_name")

    cosmo_keys = render_dynamic_list(
        "cosmo_keys", "ext_cosmo_keys", ["Omega_c", "sigma8"], cast_fn=str,
    )

    field_statistic = st.checkbox("field_statistic", value=True, key="ext_field_stat")
    power_statistic = st.checkbox("power_statistic", value=True, key="ext_power_stat")
    ddof = st.number_input("ddof", min_value=0, value=0, key="ext_ddof")
    enable_x64 = st.checkbox("enable_x64", key="ext_enable_x64")

# Build command
params = {**slurm}
params.update({
    "input_dir": input_dir if source == "Local directory" else "test_fli_samples",
    "repo_id": repo_id,
    "config": config,
    "truth_parquet": truth_parquet,
    "output_file": output_file,
    "set_name": set_name,
    "cosmo_keys": cosmo_keys,
    "field_statistic": field_statistic,
    "power_statistic": power_statistic,
    "ddof": ddof,
    "enable_x64": enable_x64,
})
cmd = build_command("extract", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
