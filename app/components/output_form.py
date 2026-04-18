"""Output settings form — output path, optional %placeholder% name, profiling.

The launcher resolves ``%constraint%``, ``%mesh_size%``, ``%box_size%``,
``%nb_steps%``, ``%omega_c%``, ``%sigma8%``, ``%seed%``, ``%lpt_order%`` in
any payload value containing them — so both ``output`` and ``name`` here can
carry those tokens and will be substituted before the script runs.
"""
from __future__ import annotations

import streamlit as st

DEFAULT_NAME_TEMPLATE = (
    "%constraint%_M%mesh_size%_B%box_size%_STEPS%nb_steps%"
    "_c%omega_c%_S8%sigma8%_s%seed%"
)


def render_output_form(
    prefix: str = "",
    defaults: dict | None = None,
    show_name: bool = True,
    profile: bool = True,
    default_output_dir: str = "results",
) -> dict:
    """Render the output settings form.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.
    show_name:
        When True, show a name template input (becomes ``--name``). Supports
        ``%placeholder%`` tokens resolved by the launcher.
    profile:
        When True, show the profile checkbox and iterations input.
    default_output_dir:
        Default value for the output path text input.

    Returns
    -------
    dict with keys:
        output, and optionally name, perf, iterations.
    """
    defaults = defaults or {}
    result: dict = {}

    with st.container(border=True):
        st.subheader("Output Settings")

        output = st.text_input(
            "output",
            value=defaults.get("output", default_output_dir),
            key=f"{prefix}output",
            help=(
                "Output path (parquet or directory). Supports %placeholder% "
                "tokens resolved by fli-launcher."
            ),
        )
        result["output"] = output

        if show_name:
            _nt_key = f"{prefix}name"
            if _nt_key not in st.session_state:
                st.session_state[_nt_key] = defaults.get("name", DEFAULT_NAME_TEMPLATE)
            nt_col, nt_btn = st.columns([3, 1])
            with nt_col:
                nt = st.text_input(
                    "name template",
                    key=_nt_key,
                    help=(
                        "Placeholders: %constraint%, %mesh_size%, %box_size%, "
                        "%nb_steps%, %omega_c%, %sigma8%, %seed%, %lpt_order%"
                    ),
                )
            with nt_btn:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Default", key=f"{prefix}name_reset"):
                    st.session_state[_nt_key] = DEFAULT_NAME_TEMPLATE
                    st.rerun()
            result["name"] = nt.strip() or None

        if profile:
            prof_col, iter_col = st.columns([2, 1])
            with prof_col:
                perf = st.checkbox(
                    "Profile",
                    value=bool(defaults.get("perf", False)),
                    key=f"{prefix}profile",
                    help="Enables --perf benchmarking (per-job mode only).",
                )
            with iter_col:
                iterations = None
                if perf:
                    iterations = st.number_input(
                        "Iter",
                        min_value=1,
                        value=int(defaults.get("iterations", 3)),
                        key=f"{prefix}iterations",
                    )
            result["perf"] = perf
            result["iterations"] = iterations if perf else None

    return result


def render_output_sample_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Render the output settings form for the Samples page.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.

    Returns
    -------
    dict with keys: output_dir.
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Output Settings")

        output_dir = st.text_input(
            "output_dir",
            value=defaults.get("output_dir", "test_fli_samples"),
            key=f"{prefix}output_dir",
        )

    return {"output_dir": output_dir}


def render_infer_config_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Render the inference configuration form for the Full Field Inference page.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.

    Returns
    -------
    dict with keys:
        observable_dir, observable, output_dir, chain_index, seed,
        adjoint, checkpoints, num_warmup, num_samples, batch_count,
        sampler, backend, sigma_e, init_cosmo.
    """
    from pathlib import Path

    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Inference Config")

        observable_path = st.text_input(
            "observable path",
            value=defaults.get("observable_path", "observables/BORN_SMALL.parquet"),
            key=f"{prefix}obs_path",
            help="Full path to the observable parquet file",
        )
        _obs_p = (
            Path(observable_path)
            if observable_path
            else Path("observables/BORN_SMALL.parquet")
        )
        observable_dir = str(_obs_p.parent)
        observable = _obs_p.name

        output_dir = st.text_input(
            "output_dir",
            value=defaults.get("output_dir", "results/inference_runs"),
            key=f"{prefix}output_dir",
        )

        ci_col, seed_col = st.columns(2)
        with ci_col:
            chain_index = st.number_input(
                "chain index",
                min_value=0,
                value=int(defaults.get("chain_index", 0)),
                key=f"{prefix}chain_index",
            )
        with seed_col:
            seed = st.number_input(
                "seed",
                min_value=0,
                value=int(defaults.get("seed", 0)),
                key=f"{prefix}seed",
            )

        adjoint = st.selectbox(
            "adjoint",
            ["checkpointed", "recursive"],
            key=f"{prefix}adjoint",
        )
        checkpoints = st.number_input(
            "checkpoints",
            min_value=1,
            value=int(defaults.get("checkpoints", 10)),
            key=f"{prefix}checkpoints",
        )

        wm_col, ns_col = st.columns(2)
        with wm_col:
            num_warmup = st.number_input(
                "num_warmup",
                min_value=0,
                value=int(defaults.get("num_warmup", 1)),
                key=f"{prefix}num_warmup",
            )
        with ns_col:
            num_samples = st.number_input(
                "num_samples",
                min_value=1,
                value=int(defaults.get("num_samples", 1)),
                key=f"{prefix}num_samples",
            )

        batch_count = st.number_input(
            "batch_count",
            min_value=1,
            value=int(defaults.get("batch_count", 2)),
            key=f"{prefix}batch_count",
        )

        sm_col, be_col = st.columns(2)
        with sm_col:
            sampler = st.selectbox(
                "sampler",
                ["NUTS", "HMC", "MCLMC"],
                key=f"{prefix}sampler",
            )
        with be_col:
            backend = st.selectbox(
                "backend",
                ["numpyro", "blackjax"],
                index=1,
                key=f"{prefix}backend",
            )

        sigma_e = st.number_input(
            "sigma_e",
            value=float(defaults.get("sigma_e", 0.26)),
            format="%.4f",
            key=f"{prefix}sigma_e",
        )
        init_cosmo = st.checkbox(
            "init_cosmo",
            value=bool(defaults.get("init_cosmo", False)),
            key=f"{prefix}init_cosmo",
            help="Warm-start cosmology from observable",
        )

    return {
        "observable_dir": observable_dir,
        "observable": observable,
        "output_dir": output_dir,
        "chain_index": chain_index,
        "seed": seed,
        "adjoint": adjoint,
        "checkpoints": checkpoints,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "batch_count": batch_count,
        "sampler": sampler,
        "backend": backend,
        "sigma_e": sigma_e,
        "init_cosmo": init_cosmo,
    }


def render_2pcf_config_form(
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Render the I/O configuration form for the 2PCF Inference page.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.

    Returns
    -------
    dict with keys: observable, path, chain_index, seed.
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("2PCF Input/Output Config")

        observable = st.text_input(
            "observable path",
            value=defaults.get(
                "observable", "observables/BORN_SMALL_spectra.parquet"
            ),
            key=f"{prefix}observable",
            help="Parquet catalog containing a PowerSpectrum field with observed C_ell.",
        )
        path = st.text_input(
            "output path",
            value=defaults.get("path", "results/2pcf_inference"),
            key=f"{prefix}output_path",
        )

        ci_col, seed_col = st.columns(2)
        with ci_col:
            chain_index = st.number_input(
                "chain index",
                min_value=0,
                value=int(defaults.get("chain_index", 0)),
                key=f"{prefix}chain_index",
            )
        with seed_col:
            seed = st.number_input(
                "seed",
                min_value=0,
                value=int(defaults.get("seed", 0)),
                key=f"{prefix}seed",
            )

    return {
        "observable": observable,
        "path": path,
        "chain_index": chain_index,
        "seed": seed,
    }
