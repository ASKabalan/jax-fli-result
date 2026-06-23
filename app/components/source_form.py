"""Catalog source form — mirrors parser.add_source_args."""
from __future__ import annotations

import streamlit as st


def render_source_form(
    defaults: dict | None = None,
    prefix: str = "",
    multi: bool = False,
    title: str = "Source",
) -> dict:
    """Catalog source: a local parquet glob (--input) OR a HuggingFace repo (--repo + --data-files).

    Returns ``None`` for the unused side so the command builder emits only the chosen flags. Mirrors
    ``jax_fli.scripts.parser.add_source_args`` (same ``prefix`` / ``multi``); reusable across the
    post-processing, inference and extraction views. ``multi=True`` makes the local ``input`` and the
    HF ``data_files`` repeatable (one pattern per line) and returns them as LISTS — used by fli-extract
    where each pattern is one MCMC chain. The returned keys are always ``input`` / ``repo`` /
    ``data_files``; a caller needing a second source remaps them (e.g. fli-infer's ``ic_*``).
    """
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader(title)

        source = st.radio(
            "Source",
            ["local glob", "HuggingFace repo"],
            index=1 if defaults.get("repo") else 0,
            horizontal=True,
            key=f"{prefix}source_mode",
            help="Read from local parquet files, or stream them from a HuggingFace dataset repo.",
        )

        input_path = repo = data_files = None
        if source == "local glob":
            if multi:
                raw = st.text_area(
                    "input patterns (one per line — each line is one chain)",
                    value=defaults.get(
                        "input", "test_fli_samples/chain_0/samples/*.parquet"
                    ),
                    key=f"{prefix}input",
                    help="One parquet glob (or a root dir) per line; each line is a separate chain.",
                )
                input_path = [
                    ln.strip() for ln in raw.splitlines() if ln.strip()
                ] or None
            else:
                input_path = (
                    st.text_input(
                        "input (file or glob)",
                        value=defaults.get("input", "results/cosmology_runs/*.parquet"),
                        key=f"{prefix}input",
                        help="A parquet path or glob, e.g. 'results/*.parquet'.",
                    )
                    or None
                )
        else:
            repo = (
                st.text_input(
                    "repo",
                    value=defaults.get("repo", "ASKabalan/jax-fli-experiments"),
                    key=f"{prefix}repo",
                    help="HuggingFace dataset repo id.",
                )
                or None
            )
            if multi:
                raw = st.text_area(
                    "data_files patterns (one per line — each line is one chain)",
                    value=defaults.get(
                        "data_files", "00-cosmogrid/chain_0/**/*.parquet"
                    ),
                    key=f"{prefix}data_files",
                    help="One glob inside the repo per line; each line is a separate chain.",
                )
                data_files = [
                    ln.strip() for ln in raw.splitlines() if ln.strip()
                ] or None
            else:
                data_files = (
                    st.text_input(
                        "data_files (glob within repo)",
                        value=defaults.get(
                            "data_files", "00-cosmogrid/density/*.parquet"
                        ),
                        key=f"{prefix}data_files",
                        help="Glob of parquet files inside the repo, e.g. '01-resolution/density/*.parquet'.",
                    )
                    or None
                )

        return {
            "input": input_path or None,
            "repo": repo or None,
            "data_files": data_files or None,
        }
