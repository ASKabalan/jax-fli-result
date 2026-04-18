"""Extract page — mirrors `fli-launch extract`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.misc_forms import render_extract_form
from app.components.slurm_form import render_slurm_form
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Extract")

slurm = render_slurm_form(prefix="ext_", show_tasks_per_node=False)
extract = render_extract_form(prefix="ext_")

# Build command — keys mirror _SUBCOMMAND_SPECS["extract"] in command_builder.py
params = {**slurm}
params.update(
    {
        "path": extract["input_dir"] if extract["source"] == "Local directory" else None,
        "repo_id": extract["repo_id"],
        "config": extract["config"],
        "set_name": extract["set_name"],
        "truth": extract["truth_parquet"],
        "output": extract["output_file"],
        "cosmo_keys": extract["cosmo_keys"],
        "field_statistic": extract["field_statistic"],
        "power_statistic": extract["power_statistic"],
        "ddof": extract["ddof"],
        "enable_x64": extract["enable_x64"],
    }
)
cmd = build_command("extract", params)
st.divider()
st.subheader("Generated command")
st.code(cmd, language="bash")
