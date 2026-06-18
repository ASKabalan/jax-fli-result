# Form ↔ CLI args mapping

Every Streamlit view emits **one** `fli-launcher` command per submit:

```
fli-launcher <SLURM flags> -- fli-<subcommand> <script flags>
```

Sweeps are no longer supported end-to-end — one invocation = one job. Parameter
scans are the user's responsibility (shell loop, external orchestrator, etc.).

This document enumerates, for each view, the forms used and the CLI flags each
form emits. Flag names match `src/jax_fli/scripts/parser.py` in `jax-fli`; the
spec tables in `app/components/command_builder.py` are the source of truth on
the UI side.

The **SLURM / cluster block** is identical for every view (mirrors
`parser.add_slurm_args`):

- `slurm_form` → `--mode`, `--account`, `--constraint`, `--nodes`,
  `--gpus-per-node`, `--cpus-per-node`, `--tasks-per-node` (optional),
  `--qos`, `--time-limit`, `--output-logs`, `--slurm-script`, `--pdim`

Launcher-side constraints:

- `gpus_per_node * nodes == prod(pdim)` — validated before dispatch.
- `--nodes` and `--pdim` are forwarded to the entry-script command after `--`
  so JAX sharding sees the same world shape.
- `--tasks-per-node` is shown only for CPU jobs (e.g. Dorian RT). GPU jobs
  derive one task per GPU from `--gpus-per-node`.

### Template substitution

Any payload value containing ``%token%`` is resolved by the launcher before
dispatch. Supported tokens:

    %constraint%   — from launcher --constraint
    %mesh_size%    — --mesh-size values joined with 'x' (e.g. 64x64x64)
    %box_size%     — --box-size  values joined with 'x'
    %nb_steps%     — --nb-steps
    %omega_c%      — --Omega-c
    %sigma8%       — --sigma8
    %seed%         — --seed
    %lpt_order%    — --lpt-order

This is typically used in ``--output`` and ``--name`` (the launcher also uses
``--name``, when present, as the SLURM job name). After resolution, the
launcher writes ``<output_dir>/args.log`` with the resolved command.
``<output_dir>`` is the parent of ``--output`` when it looks like a file
(``.parquet``, ``.h5``, ...) or the value of ``--output`` / ``--path`` when
it looks like a directory. ``fli-spectra`` has no output flag and is skipped.

---

## `1_Simulate.py` → `fli-simulate`

Forms:

- `slurm_form` (shared SLURM block, `show_pdim=True`).
- `simulation_settings_form` (`show_output_target=True`) →
  `--mesh-size`, `--box-size`, `--observer-position`, `--halo-multiplier`,
  `--seed`, `--scheme`, `--paint-nside`, `--kernel-width-arcmin`,
  `--enable-x64`, one-of(`--nside` | `--flatsky-npix --field-size` | `--density`).
  *Constraint: the four output-target flags are mutually exclusive; the form
  emits exactly one of them based on the radio selection.*
- `integration_form` → `--sim-mode`, `--lpt-order`, `--nb-steps`, `--t0`,
  `--t1`, `--interp`, `--solver`, `--time-stepping`, `--shell-spacing`,
  `--dealiased`, `--exact-growth`, `--gradient-order`, `--laplace-fd`,
  `--density-widths`, one-of(`--nb-shells` | `--ts` | `--ts-near --ts-far`),
  `--min-width`, `--drift-on-lightcone`.
  *Constraint: `--ts` and (`--ts-near`, `--ts-far`) are mutually exclusive with
  each other and with `--nb-shells`.*
- `cosmo_form` → `--Omega-c`, `--sigma8`, `--h`, `--Omega-b`, `--Omega-k`,
  `--Omega-nu`, `--w0`, `--wa`, `--n-s` (all scalar — gridding removed).
- `output_form` → `--output`, `--name`, `--perf`, `--iterations`.
  *Both `--output` and `--name` accept ``%placeholder%`` tokens (see
  "Template substitution" above).*

---

## `2_Samples.py` → `fli-samples`

Forms:

- `slurm_form` (shared).
- `simulation_settings_form` (`show_nside=True`) → same simulation-settings
  flags as above, with `--nside` always emitted (samples are always spherical).
- `integration_form` → same as Simulate.
- `prior_cosmo_form` (`show_ic=True`) → `--sample`, `--prior-omega-c`,
  `--prior-sigma8`, `--prior-h`, `--prior-ic-gaussian`, `--initial-condition`.
  *Constraint: each prior range is emitted only when the corresponding "Fixed"
  checkbox is off; the `--sample` list is derived from those toggles.*
- Samples settings (inline) → `--model`, `--num-samples`, `--sigma-e`,
  `--batch-id`.

---

## `3_FullFieldInference.py` → `fli-infer`

Forms:

- `slurm_form` (shared).
- `simulation_settings_form` (scalar) — same surface as Simulate/Samples.
- `integration_form` → same as Simulate.
- `prior_cosmo_form` (`show_ic=True`) → same priors as Samples.
- `output_form` / `render_infer_config_form` → `--path`, `--observable`,
  `--init-cosmo`, `--initial-condition`, `--sigma-e`, `--adjoint`,
  `--checkpoints`, `--num-warmup`, `--num-samples`, `--batch-count`,
  `--sampler`, `--backend`, `--no-progress-bar`.

---

## `4_2PCFInference.py` → `fli-2pcf`

Forms:

- `slurm_form` (shared, `show_pdim=False` — CPU/single-GPU job).
- `render_2pcf_observable_form` → `--observable`, one-of(`--nside` |
  `--flatsky-npix --field-size`), `--lmax`, `--f-sky`, `--sigma-e`,
  `--nonlinear-fn`, `--num-warmup`, `--num-samples`, `--batch-count`,
  `--sampler`, `--backend`, `--enable-x64`.
  *Constraint: `--nside` and (`--flatsky-npix`, `--field-size`) are mutually
  exclusive.*
- `render_2pcf_config_form` → `--path`, `--chain-index`, `--seed`.
- `prior_cosmo_form` (`show_ic=False`) → `--sample`, `--prior-omega-c`,
  `--prior-sigma8`, `--prior-h` (no IC priors — 2PCF is cosmo-only).

---

## `5_Extract.py` → `fli-extract`

Forms:

- `slurm_form` (shared, `show_tasks_per_node=False`).
- `render_extract_form` → one-of(`--path` | `--repo-id --config`), `--truth`,
  `--set-name`, `--output`, `--cosmo-keys`, `--field-statistic`,
  `--power-statistic`, `--ddof`, `--enable-x64`.
  *Constraint: `--path` (local directory) and `--repo-id` (HuggingFace dataset)
  are mutually exclusive — the radio selection enforces this.*

---

## `6_Born_RT.py` → `fli-born-rt`

Forms:

- `slurm_form` (shared).
- `lensing_form` → `--nz-shear`, `--min-z`, `--max-z`, `--n-integrate`.
- Inline → `--input`, `--output`, `--enable-x64`.

---

## `7_Dorian_RT.py` → `fli-dorian-rt`

Forms:

- `slurm_form` (shared, `show_tasks_per_node=True`, `show_pdim=False`) —
  Dorian is a CPU job that uses MPI tasks, not a JAX sharding grid.
- `lensing_form` → `--nz-shear`, `--min-z`, `--max-z`, `--n-integrate`.
- Inline → `--input`, `--output`, `--rt-interp`, `--no-parallel-transport`.

---

## `8_Spectra.py` → `fli-spectra`

Forms:

- `slurm_form` (shared, `show_pdim=False` — CPU job).
- Inline widgets → **positional** `folder`, then `--regex`, `--recursive`,
  `--force-regen`, `--normalization`, `--ell-edges`, `--lmax`, `--kmax`,
  `--method`, `--kedges`, `--multipoles`, `--los`, `--batch-size`,
  `--enable-x64`.
  *Note: `--ell-edges`, `--kedges`, `--multipoles`, `--los` are vector-valued
  flags and still use `dynamic_list` — these are not sweeps, they are
  multi-element arguments to a single call.*
  *Note: `folder` is emitted as a positional argument (no `--folder` prefix).*
