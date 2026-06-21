"""Summary Statistics page — mirrors `fli-launcher -- fli-summary-stats`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.slurm_form import render_slurm_form
from app.components.styled_container import inject_custom_css
from app.components.summary_stats_form import (
    render_summary_stats_common_form,
    render_summary_stats_density_form,
    render_summary_stats_flat_form,
    render_summary_stats_mask_form,
    render_summary_stats_scan_form,
    render_summary_stats_spherical_form,
)

inject_custom_css()
st.title("Summary Statistics")

st.markdown(
    "Compute summary statistics from lightcone/density parquet catalogs.\n\n"
    "Scans a folder for `.parquet` files and computes angular C_ell (flat or spherical) "
    "or 3D P(k) depending on the field type found. Spherical maps support an apodized "
    "footprint mask (observer position / DES Y3 / file)."
)

cmd_placeholder = st.empty()

c1, c2 = st.columns([1, 2])

with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 0, "nodes": 1, "cpus_per_node": 8},
        prefix="ss_slurm_",
        show_pdim=False,
        show_tasks_per_node=False,
    )

with c2:
    scan = render_summary_stats_scan_form(prefix="ss_")
    flat = render_summary_stats_flat_form(prefix="ss_")
    spherical = render_summary_stats_spherical_form(prefix="ss_")
    density = render_summary_stats_density_form(prefix="ss_")
    mask = render_summary_stats_mask_form(prefix="ss_")
    common = render_summary_stats_common_form(prefix="ss_")

params = {
    **slurm,
    **scan,
    **flat,
    **spherical,
    **density,
    **mask,
    **common,
}

cmd = build_command("summary-stats", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
