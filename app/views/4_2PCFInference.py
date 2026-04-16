"""2PCF Inference page — mirrors `fli-launcher 2pcf` (fli-2pcf)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.command_builder import build_command
from app.components.lensing_form import render_lensing_form
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
# c1 — SLURM + 2PCF Config
# ─────────────────────────────────────────────────────────────────────────────
with c1:
    slurm = render_slurm_form(
        defaults={"gpus_per_node": 1, "nodes": 1, "pdim": [1, 1]},
        prefix="tpcf_slurm_",
        show_pdim=False,
    )

    with st.container(border=True):
        st.subheader("2PCF Config")

        observable = st.text_input(
            "observable path",
            value="observables/BORN_SMALL_spectra.parquet",
            key="tpcf_observable",
            help="Parquet Catalog containing a PowerSpectrum field with observed C_ell.",
        )
        output_path = st.text_input(
            "output path",
            value="results/2pcf_inference",
            key="tpcf_output_path",
        )

        ci_col, seed_col = st.columns(2)
        with ci_col:
            chain_index = st.number_input(
                "chain index", min_value=0, value=0, key="tpcf_chain_index"
            )
        with seed_col:
            seed = st.number_input("seed", min_value=0, value=0, key="tpcf_seed")

        st.markdown("**Geometry**")
        geom_mode = st.radio(
            "geometry",
            ["Spherical (nside)", "Flat sky"],
            horizontal=True,
            key="tpcf_geom_mode",
        )
        nside = flatsky_npix = field_size = None
        if geom_mode == "Spherical (nside)":
            nside = st.number_input("nside", min_value=1, value=64, key="tpcf_nside")
        else:
            fp1, fp2 = st.columns(2)
            with fp1:
                _fp_h = st.number_input(
                    "H (pixels)", min_value=1, value=512, key="tpcf_fp_h"
                )
            with fp2:
                _fp_w = st.number_input(
                    "W (pixels)", min_value=1, value=512, key="tpcf_fp_w"
                )
            flatsky_npix = [_fp_h, _fp_w]
            ff1, ff2 = st.columns(2)
            with ff1:
                _ff_h = st.number_input(
                    "H (deg)", min_value=1, value=10, key="tpcf_ff_h"
                )
            with ff2:
                _ff_w = st.number_input(
                    "W (deg)", min_value=1, value=10, key="tpcf_ff_w"
                )
            field_size = [_ff_h, _ff_w]

        lmax_col, fsky_col = st.columns(2)
        with lmax_col:
            lmax = st.number_input("lmax", min_value=1, value=2047, key="tpcf_lmax")
        with fsky_col:
            f_sky = st.number_input(
                "f_sky",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                format="%.3f",
                key="tpcf_f_sky",
            )

        sigma_e_col, nl_col = st.columns(2)
        with sigma_e_col:
            sigma_e = st.number_input(
                "sigma_e", value=0.26, format="%.4f", key="tpcf_sigma_e"
            )
        with nl_col:
            nonlinear_fn = st.selectbox(
                "nonlinear_fn",
                ["halofit", "linear"],
                key="tpcf_nonlinear_fn",
            )

        st.markdown("**MCMC**")
        wm_col, ns_col = st.columns(2)
        with wm_col:
            num_warmup = st.number_input(
                "num_warmup", min_value=0, value=100, key="tpcf_num_warmup"
            )
        with ns_col:
            num_samples = st.number_input(
                "num_samples", min_value=1, value=500, key="tpcf_num_samples"
            )

        batch_count = st.number_input(
            "batch_count", min_value=1, value=10, key="tpcf_batch_count"
        )

        sm_col, be_col = st.columns(2)
        with sm_col:
            sampler = st.selectbox(
                "sampler", ["NUTS", "HMC", "MCLMC"], key="tpcf_sampler"
            )
        with be_col:
            backend = st.selectbox(
                "backend", ["numpyro", "blackjax"], index=1, key="tpcf_backend"
            )

        enable_x64 = st.checkbox("enable_x64", value=False, key="tpcf_enable_x64")

# ─────────────────────────────────────────────────────────────────────────────
# c2 — Lensing
# ─────────────────────────────────────────────────────────────────────────────
with c2:
    lensing = render_lensing_form(prefix="tpcf_")

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

params = {**slurm, **lensing}
params.update(
    {
        "observable": observable,
        "path": output_path,
        "nside": nside,
        "flatsky_npix": flatsky_npix,
        "field_size": field_size,
        "lmax": lmax,
        "f_sky": f_sky,
        "sigma_e": sigma_e,
        "nonlinear_fn": nonlinear_fn,
        "chain_index": chain_index,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "batch_count": batch_count,
        "sampler": sampler,
        "backend": backend,
        "sample": sample if sample else ["cosmo"],
        "prior_omega_c": prior_omega_c,
        "prior_sigma8": prior_sigma8,
        "prior_h": prior_h,
        "seed": seed,
        "enable_x64": enable_x64,
    }
)

cmd = build_command("2pcf", params)

with cmd_placeholder:
    st.subheader("Generated command")
    st.code(cmd, language="bash")
