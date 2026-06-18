# jax-fli-launcher

**Streamlit GUI for launching [jax-fli](https://github.com/ASKabalan/jax-fli) simulation jobs.**

A point-and-click front end over the `fli-*` command-line tools shipped by `jax-fli`. Each
view builds a single `fli-launcher` command (SLURM block + `fli-<subcommand>` flags) with a
live preview, and dispatches it in `dryrun` / `local` / `sbatch` mode. An in-app **Analysis**
tab loads catalogs (incl. from the HuggingFace dataset) and plots density / spherical /
kappa diagnostics and summary statistics.

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/) and a committed `uv.lock`
for reproducibility. Dependency groups are **not** installed by default:

```bash
uv sync                  # app + jax-fli[catalog,plot] + streamlit/plotly/matplotlib
uv sync --group dev      # + dev tooling: ruff, pyright, prek, toml-sort
```

> **Note:** `jax-fli` (and its `jaxpm` / `jax-cosmo` / `s2fft` forks) are pinned to git
> branches via `[tool.uv.sources]`, which **only `uv` reads** — installing with plain `pip`
> would fetch the upstream PyPI releases instead of the forks the Analysis tab needs, so
> use `uv`.

## Running

```bash
uv run streamlit run app/Home.py
```

## Structure

- `app/Home.py` — navigation router / entry point
- `app/views/<N>_<Name>.py` — one page per `fli-*` subcommand (Simulate, Samples,
  Inference, Extract, Born/Dorian RT, Spectra) plus the Analysis tab (`9_Analysis.py` +
  `app/views/_9_Analysis/`)
- `app/components/` — reusable form renderers and the command builder
- `MAPPING.md` — form ↔ CLI flag mapping for every view
