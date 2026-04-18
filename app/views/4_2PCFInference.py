"""2PCF Inference page — mirrors `fli-launcher 2pcf` (fli-2pcf)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.misc_forms import render_2pcf_observable_form
from app.components.output_form import render_2pcf_config_form
from app.components.prior_cosmo_form import render_prior_cosmo_form
from app.components.slurm_form import render_slurm_form
from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("2PCF Inference")

# ── TOP ROW: description ──────────────────────────────────────────────────────
st.markdown(
    "Run 2-point-function (power-spectrum level) MCMC inference.\n\n"
    "Uses a Knox-formula Gaussian likelihood on pre-computed C_ell — "
    "orders of magnitude faster than full-field inference."
)

# ── MIDDLE: command preview placeholder ──────────────────────────────────────
cmd_placeholder = st.empty()

# ── 3 columns ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# c1 — SLURM + 2PCF I/O Config
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 1, "nodes": 1, "pdim": [1, 1]},
        prefix="tpcf_slurm_",
        show_pdim=False,
    )
    tpcf_io = render_2pcf_config_form(prefix="tpcf_")
# ─────────────────────────────────────────────────────────────────────────────
# c2 — 2PCF Observable Settings
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    tpcf_obs = render_2pcf_observable_form(prefix="tpcf_")
# ─────────────────────────────────────────────────────────────────────────────
# c3 — Prior Cosmology (no IC for 2PCF)
# ─────────────────────────────────────────────────────────────────────────────
with c3:
    cosmo = render_prior_cosmo_form(prefix="tpcf_", show_ic=False)

# ── Build command ─────────────────────────────────────────────────────────────
prior_omega_c = cosmo.get("prior_omega_c") or [0.1, 0.5]
prior_sigma8 = cosmo.get("prior_sigma8") or [0.6, 1.0]
prior_h = cosmo.get("prior_h") or [0.5, 0.9]
sample = cosmo.get("sample") or ["cosmo"]

params = {**slurm}
params.update(
    {
        "observable": tpcf_io["observable"],
        "path": tpcf_io["path"],
        "chain_index": tpcf_io["chain_index"],
        "seed": tpcf_io["seed"],
        "nside": tpcf_obs["nside"],
        "flatsky_npix": tpcf_obs["flatsky_npix"],
        "field_size": tpcf_obs["field_size"],
        "lmax": tpcf_obs["lmax"],
        "f_sky": tpcf_obs["f_sky"],
        "sigma_e": tpcf_obs["sigma_e"],
        "nonlinear_fn": tpcf_obs["nonlinear_fn"],
        "num_warmup": tpcf_obs["num_warmup"],
        "num_samples": tpcf_obs["num_samples"],
        "batch_count": tpcf_obs["batch_count"],
        "sampler": tpcf_obs["sampler"],
        "backend": tpcf_obs["backend"],
        "enable_x64": tpcf_obs["enable_x64"],
        "sample": sample if sample else ["cosmo"],
        "prior_omega_c": prior_omega_c,
        "prior_sigma8": prior_sigma8,
        "prior_h": prior_h,
    }
)

cmd = build_command("2pcf", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
