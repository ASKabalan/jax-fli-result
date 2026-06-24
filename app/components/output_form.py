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


def _reset_name_template(nt_key: str) -> None:
    """Reset the name-template field (callback runs before rerun)."""
    st.session_state[nt_key] = DEFAULT_NAME_TEMPLATE


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
                st.button(
                    "Default",
                    key=f"{prefix}name_reset",
                    on_click=_reset_name_template,
                    args=(_nt_key,),
                )
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

        shells_per_file = st.number_input(
            "shells_per_file",
            min_value=0,
            value=int(defaults.get("shells_per_file", 0)),
            key=f"{prefix}shells_per_file",
            help=(
                "Stream a multi-shell lightcone N shells per parquet file (0 = one file). "
                "When >=1, output is treated as a directory."
            ),
        )
        result["shells_per_file"] = int(shells_per_file)

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
        path (output dir), chain_index, seed, adjoint, checkpoints, num_warmup,
        num_samples, batch_count, sampler, max_num_doublings, target_accept,
        mclmc_desired_energy_var, mclmc_init_step_size_scale, sigma_e, init_cosmo,
        no_progress_bar. The observable / initial-condition sources are rendered
        separately by render_source_form in the view (fli-infer --input / --ic-input).
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Inference Config")

        path = st.text_input(
            "output_dir",
            value=defaults.get("output_dir", "results/inference_runs"),
            key=f"{prefix}output_dir",
            help="Output directory for MCMC checkpoints and catalogs (fli-infer --path).",
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

        sampler = st.selectbox(
            "sampler",
            ["NUTS", "MCLMC"],
            key=f"{prefix}sampler",
        )

        # NUTS tuning
        mnd_col, ta_col = st.columns(2)
        with mnd_col:
            max_num_doublings = st.number_input(
                "max_num_doublings",
                min_value=1,
                value=int(defaults.get("max_num_doublings", 10)),
                key=f"{prefix}max_num_doublings",
                help="NUTS leapfrog trajectory doubling depth.",
            )
        with ta_col:
            target_accept = st.number_input(
                "target_accept",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults.get("target_accept", 0.8)),
                format="%.2f",
                key=f"{prefix}target_accept",
                help="NUTS window-adaptation target acceptance rate.",
            )

        # MCLMC tuning
        mev_col, mss_col = st.columns(2)
        with mev_col:
            mclmc_desired_energy_var = st.number_input(
                "mclmc_desired_energy_var",
                min_value=0.0,
                value=float(defaults.get("mclmc_desired_energy_var", 1e-3)),
                format="%.1e",
                key=f"{prefix}mclmc_desired_energy_var",
                help="MCLMC desired energy variance for L/step_size tuning.",
            )
        with mss_col:
            mclmc_init_step_size_scale = st.number_input(
                "mclmc_init_step_size_scale",
                min_value=0.0,
                value=float(defaults.get("mclmc_init_step_size_scale", 1e-4)),
                format="%.1e",
                key=f"{prefix}mclmc_init_step_size_scale",
                help="MCLMC initial step size = sqrt(total_dim) * scale.",
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
        no_progress_bar = st.checkbox(
            "no_progress_bar",
            value=bool(defaults.get("no_progress_bar", False)),
            key=f"{prefix}no_progress_bar",
            help="Suppress tqdm progress bars (fli-infer --no-progress-bar).",
        )

    return {
        "path": path,
        "chain_index": chain_index,
        "seed": seed,
        "adjoint": adjoint,
        "checkpoints": checkpoints,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "batch_count": batch_count,
        "sampler": sampler,
        "max_num_doublings": max_num_doublings,
        "target_accept": target_accept,
        "mclmc_desired_energy_var": mclmc_desired_energy_var,
        "mclmc_init_step_size_scale": mclmc_init_step_size_scale,
        "sigma_e": sigma_e,
        "init_cosmo": init_cosmo,
        "no_progress_bar": no_progress_bar,
    }
