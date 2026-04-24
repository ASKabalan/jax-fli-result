"""Spectra page — mirrors `fli-launcher spectra` (fli-spectra)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.slurm_form import render_slurm_form
from app.components.spectra_form import (
    render_spectra_common_form,
    render_spectra_density_form,
    render_spectra_flat_form,
    render_spectra_scan_form,
    render_spectra_spherical_form,
)
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Spectra")

st.markdown(
    "Compute power spectra from lightcone/density parquet catalogs.\n\n"
    "Scans a folder for `.parquet` files and computes angular C_ell (flat or spherical) "
    "or 3D P(k) depending on the field type found."
)

cmd_placeholder = st.empty()

c1, c2 = st.columns([1, 2])

with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 0, "nodes": 1, "cpus_per_node": 8},
        prefix="spec_slurm_",
        show_pdim=False,
        show_tasks_per_node=False,
    )

with c2:
    scan = render_spectra_scan_form(prefix="spec_")
    flat = render_spectra_flat_form(prefix="spec_")
    spherical = render_spectra_spherical_form(prefix="spec_")
    density = render_spectra_density_form(prefix="spec_")
    common = render_spectra_common_form(prefix="spec_")

params = {
    **slurm,
    **scan,
    **flat,
    **spherical,
    **density,
    **common,
}

cmd = build_command("spectra", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
