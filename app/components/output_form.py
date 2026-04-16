"""Output settings form — output_dir, name template, profiling."""
from __future__ import annotations

import streamlit as st

try:
    from app.components.command_builder import DEFAULT_NAME_TEMPLATE
except ImportError:
    DEFAULT_NAME_TEMPLATE = "%constraint%_cosmo_M%mesh_size%_B%box_size%_STEPS%nb_steps%_c%omega_c%_S8%sigma8%_s%seed%"


def render_output_form(
    prefix: str = "",
    defaults: dict | None = None,
    name_template: bool = True,
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
    name_template:
        When True, show the name-template input with reset button.
    profile:
        When True, show the profile checkbox and iterations input.
    default_output_dir:
        Default value for the output directory text input.

    Returns
    -------
    dict with keys:
        output_dir, and optionally name_template, perf, iterations.
    """
    defaults = defaults or {}
    result: dict = {}

    with st.container(border=True):
        st.subheader("Output Settings")

        output_dir = st.text_input(
            "output_dir",
            value=defaults.get("output_dir", default_output_dir),
            key=f"{prefix}output_dir",
            help="Output directory for results.",
        )
        result["output_dir"] = output_dir

        if name_template:
            _nt_key = f"{prefix}name_template"
            if _nt_key not in st.session_state:
                st.session_state[_nt_key] = defaults.get(
                    "name_template", DEFAULT_NAME_TEMPLATE
                )
            _nt_col, _nt_btn = st.columns([3, 1])
            with _nt_col:
                nt = st.text_input(
                    "Name template",
                    key=_nt_key,
                    help=(
                        "Placeholders: %constraint%, %mesh_size%, %box_size%, "
                        "%nb_steps%, %omega_c%, %sigma8%, %seed%"
                    ),
                )
            with _nt_btn:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Default", key=f"{prefix}name_template_reset"):
                    st.session_state[_nt_key] = DEFAULT_NAME_TEMPLATE
                    st.rerun()
            result["name_template"] = nt

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
