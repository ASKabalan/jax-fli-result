"""Spectra page — mirrors `fli-launcher spectra` (fli-spectra)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.dynamic_list import render_dynamic_list
from app.components.slurm_form import render_slurm_form
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Spectra")

# ── TOP ROW: description ──────────────────────────────────────────────────────
st.markdown(
    "Compute power spectra from lightcone/density parquet catalogs.\n\n"
    "Scans a folder for `.parquet` files and computes angular C_ell (flat or spherical) "
    "or 3D P(k) depending on the field type found."
)

# ── MIDDLE: command preview placeholder ──────────────────────────────────────
cmd_placeholder = st.empty()

# ── 2-column layout: SLURM left | Spectra config right ───────────────────────
c1, c2 = st.columns([1, 2])

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM (no pdim — CPU job)
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 0, "nodes": 1, "cpus_per_node": 8},
        prefix="spec_slurm_",
        show_pdim=False,
        show_tasks_per_node=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Spectra settings
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    with st.container(border=True):
        st.subheader("Scan")

        folder = st.text_input(
            "folder",
            value="results/cosmology_runs",
            key="spec_folder",
            help="Root folder to scan for parquet files.",
        )
        regex = st.text_input(
            "regex",
            value=r".*\.parquet$",
            key="spec_regex",
            help="Regex pattern to filter filenames (default: all .parquet files).",
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            recursive = st.checkbox("recursive", value=False, key="spec_recursive")
        with sc2:
            force_regen = st.checkbox("force_regen", value=False, key="spec_force_regen")
        with sc3:
            normalization = st.selectbox(
                "normalization",
                ["global", "per_plane"],
                key="spec_normalization",
            )

    with st.container(border=True):
        st.subheader("Angular C_ell")

        st.markdown("**Flat-sky**")
        ell_edges = render_dynamic_list("ell_edges", "spec_ell_edges", [], cast_fn=float) or None

        st.markdown("**Spherical (HEALPix)**")
        use_lmax = st.checkbox("Override lmax", value=False, key="spec_use_lmax")
        lmax = None
        if use_lmax:
            lmax = st.number_input(
                "lmax", min_value=1, value=511, key="spec_lmax",
                help="Default: 3*nside-1"
            )
        method = st.selectbox(
            "SHT method", ["healpy", "jax"], key="spec_method"
        )

    with st.container(border=True):
        st.subheader("3D P(k)")

        kedges = render_dynamic_list("kedges", "spec_kedges", [], cast_fn=float) or None
        multipoles = render_dynamic_list("multipoles", "spec_multipoles", ["0"], cast_fn=int) or [0]

        lo1, lo2, lo3 = st.columns(3)
        with lo1:
            los_x = st.number_input("LOS x", value=0.0, format="%.2f", key="spec_los_x")
        with lo2:
            los_y = st.number_input("LOS y", value=0.0, format="%.2f", key="spec_los_y")
        with lo3:
            los_z = st.number_input("LOS z", value=1.0, format="%.2f", key="spec_los_z")

    with st.container(border=True):
        st.subheader("Common")
        bs_col, x64_col = st.columns(2)
        with bs_col:
            use_batch = st.checkbox("Set batch size", value=False, key="spec_use_batch")
            batch_size = None
            if use_batch:
                batch_size = st.number_input(
                    "batch_size", min_value=1, value=4, key="spec_batch_size"
                )
        with x64_col:
            enable_x64 = st.checkbox("enable_x64", value=False, key="spec_enable_x64")

# ── Build command ─────────────────────────────────────────────────────────────
params = {**slurm}
params.update(
    {
        "folder": folder,
        "regex": regex,
        "recursive": recursive,
        "force_regen": force_regen,
        "normalization": normalization,
        "ell_edges": ell_edges,
        "lmax": lmax,
        "method": method,
        "kedges": kedges,
        "multipoles": multipoles,
        "los": [los_x, los_y, los_z],
        "batch_size": batch_size,
        "enable_x64": enable_x64,
    }
)

cmd = build_command("spectra", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
