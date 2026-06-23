"""Build the CLI command string from collected parameters.

Emits:

    fli-launcher <SLURM flags> -- fli-<subcommand> <script flags>

The SLURM spec is shared across all subcommands. Each subcommand has its own script-side spec
below, composed from one fragment per ``jax_fli.scripts.parser`` builder so the form/command stays
a 1:1 mirror of the CLI. ``build_command`` always returns a single string — gridding/sweeping is
not supported (the launcher runs one job per call).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Spec vocabulary
# ---------------------------------------------------------------------------
# Each entry is (flag, type, default). ``type`` is one of:
#   str / int / float           — scalar, always emitted (unless None)
#   bool                        — emitted only when True (store_true flag)
#   list                        — --flag v1 v2 ...
#   optional_int / optional_str / optional_float / optional_list — skipped when None
# ---------------------------------------------------------------------------

_DISTRIBUTED_SPEC = [
    ("nodes", int, 1),
    ("pdim", list, [1, 1]),
]

_DISTRIBUTED_SUBCOMMANDS = {"simulate", "samples", "infer", "extract", "born-rt"}

_SLURM_SPEC = [
    ("mode", str, "dryrun"),
    ("account", str, "XXX"),
    ("constraint", "optional_str", None),
    ("nodes", int, 1),
    ("gpus-per-node", int, 4),
    ("cpus-per-node", int, 16),
    ("tasks-per-node", "optional_int", None),
    ("qos", str, "qos_gpu_h100-t3"),
    ("time-limit", str, "00:30:00"),
    ("slurm-script", "optional_str", None),
    ("output-logs", str, "SLURM_LOGS"),
    ("pdim", list, [1, 1]),
]

# ---------------------------------------------------------------------------
# Science-side spec fragments — one per jax_fli.scripts.parser builder.
# ---------------------------------------------------------------------------

# add_common_args
_COMMON_SPEC = [
    ("enable-x64", bool, False),
]

# add_integration_settings_args (NOTE: --sim-mode is fli-simulate-only, added in that spec)
_INTEGRATION_SPEC = [
    ("lpt-order", int, 2),
    ("t0", float, 0.001),
    ("t1", float, 1.0),
    ("nb-steps", int, 30),
    ("nb-shells", "optional_int", None),
    ("interp", str, "none"),
    ("solver", str, "kdk"),
    ("time-stepping", str, "a"),
    ("shell-spacing", str, "comoving"),
    ("dealiased", bool, False),
    ("exact-growth", bool, False),
    ("gradient-order", int, 1),
    ("laplace-fd", bool, False),
    ("paint-order", str, "cic"),
    ("deconvolution", bool, False),
    ("density-widths", "optional_list", None),
    ("ts", "optional_list", None),
    ("ts-near", "optional_list", None),
    ("ts-far", "optional_list", None),
    ("drift-on-lightcone", bool, False),
    ("min-width", float, 50.0),
]

# add_lensing_args (NOTE: --lensing-output is NOT here — it lives in add_forward_model_args,
# which only fli-infer / fli-samples compose. fli-simulate / fli-born-rt / fli-dorian-rt emit
# convergence only and reject --lensing-output.)
_LENSING_SPEC = [
    ("nz-shear", list, ["s3"]),
    ("min-z", float, 0.01),
    ("max-z", float, 1.5),
    ("n-integrate", int, 32),
]

# add_source_args (generic catalog source). EITHER a local glob (--input) OR a HuggingFace repo
# (--repo + --data-files); the form returns None for the unused side so only the chosen flags emit.
_SOURCE_SPEC = [
    ("input", "optional_str", None),
    ("repo", "optional_str", None),
    ("data-files", "optional_str", None),
]

# add_source_args(prefix="ic") — fli-infer's OPTIONAL initial-condition source (single, prefixed).
_IC_SOURCE_SPEC = [
    ("ic-input", "optional_str", None),
    ("ic-repo", "optional_str", None),
    ("ic-data-files", "optional_str", None),
]

# add_source_args(multi=True) — fli-extract: each pattern is one chain (input / data-files are lists).
_SOURCE_MULTI_SPEC = [
    ("input", "optional_list", None),
    ("repo", "optional_str", None),
    ("data-files", "optional_list", None),
]

# add_lensing_postproc_args (output + density→κ knobs for fli-born-rt / fli-dorian-rt). The --output
# default is supplied by the view.
_LENSING_POSTPROC_SPEC = [
    ("nside", "optional_int", None),
    ("normalization", str, "global"),
    ("output", str, "."),
]

# add_simulation_settings_args (box geometry + RNG). NOTE: --apodization-scale-deg is NOT here —
# it lives in add_forward_model_args (fli-infer / fli-samples); fli-simulate rejects it.
_SIM_SETTINGS_SPEC = [
    ("mesh-size", list, [64, 64, 64]),
    ("box-size", list, [200.0, 200.0, 200.0]),
    ("halo-multiplier", float, 0.5),
    ("observer-position", list, [0.5, 0.5, 0.5]),
    ("seed", int, 0),
]

# add_output_target_args (projection target + painting scheme)
_OUTPUT_TARGET_SPEC = [
    # Output target (mutually exclusive — only one should be set per call).
    ("nside", "optional_int", None),
    ("flatsky-npix", "optional_list", None),
    ("field-size", "optional_list", None),
    ("density", bool, False),
    ("scheme", str, "bilinear"),
    ("paint-nside", "optional_int", None),
    ("kernel-width-arcmin", "optional_float", None),
    ("kernel-width-pixels", "optional_float", None),
    ("pixel-window-deconvolution", bool, False),
]

# add_cosmo_args
_COSMO_SPEC = [
    ("h", float, 0.6774),
    ("Omega-b", float, 0.0486),
    ("Omega-k", float, 0.0),
    ("Omega-nu", float, 0.0),
    ("w0", float, -1.0),
    ("wa", float, 0.0),
    ("n-s", float, 0.9667),
    ("Omega-c", float, 0.2589),
    ("sigma8", float, 0.8159),
]

# add_prior_args
_PRIOR_SPEC = [
    ("sample", list, ["cosmo", "ic"]),
    ("prior-omega-c", list, [0.1, 0.5]),
    ("prior-sigma8", list, [0.6, 1.0]),
    ("prior-h", list, [0.5, 0.9]),
    ("prior-ic-gaussian", list, [0.0, 1.0]),
]

# add_infer_args (fli-infer's IC comes from the prefixed _IC_SOURCE_SPEC, not --initial-condition).
# NOTE: there is no --backend flag in add_infer_args — fli-infer rejects it.
_INFER_SPEC = [
    ("init-cosmo", bool, False),
    ("sigma-e", float, 0.26),
    ("num-warmup", int, 500),
    ("num-samples", int, 1000),
    ("batch-count", int, 5),
    ("adjoint", str, "checkpointed"),
    ("checkpoints", int, 10),
    ("sampler", str, "NUTS"),
    ("max-num-doublings", int, 10),
    ("target-accept", float, 0.8),
    ("mclmc-desired-energy-var", float, 1e-3),
    ("mclmc-init-step-size-scale", float, 1e-4),
    ("no-progress-bar", bool, False),
]

# add_forward_model_args (full-field likelihood, fli-infer / fli-samples only): --lensing-output
# (convergence vs shear — a forward-model concern) and --apodization-scale-deg live here, not in the
# lensing / simulation-settings groups.
_FORWARD_MODEL_SPEC = [
    ("lensing-output", str, "convergence"),
    ("mask", "optional_str", None),
    ("sigma-unobserved", float, 1e6),
    ("log-lightcone", bool, False),
    ("apodization-scale-deg", float, 1.0),
    ("map2alm-method", str, "jax"),
]

# Per-subcommand script specs (what goes AFTER the `--`).
_SUBCOMMAND_SPECS: dict[str, list] = {
    "simulate": [("sim-mode", str, "lensing")]
    + _COMMON_SPEC
    + _SIM_SETTINGS_SPEC
    + _OUTPUT_TARGET_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _COSMO_SPEC
    + [
        ("grad", "optional_str", None),
        ("output", str, "sim_output.parquet"),
        ("name", "optional_str", None),
        ("perf", bool, False),
        ("iterations", int, 5),
        ("shells-per-file", int, 0),
    ],
    "samples": _COMMON_SPEC
    + _SIM_SETTINGS_SPEC
    + _OUTPUT_TARGET_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _PRIOR_SPEC
    + _FORWARD_MODEL_SPEC
    + [
        ("path", str, "test_fli_samples"),
        ("model", str, "full"),
        ("sigma-e", float, 0.26),
        ("num-samples", int, 100),
        ("batch-id", int, 0),
        ("initial-condition", "optional_str", None),
    ],
    "infer": _COMMON_SPEC
    + _SIM_SETTINGS_SPEC
    + _OUTPUT_TARGET_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _PRIOR_SPEC
    + _INFER_SPEC
    + _FORWARD_MODEL_SPEC
    + _SOURCE_SPEC  # single-row observable (--input XOR --repo/--data-files)
    + _IC_SOURCE_SPEC  # optional initial condition (--ic-input XOR --ic-repo/--ic-data-files)
    + [
        ("path", str, "results/inference_runs"),
    ],
    "extract": _SOURCE_MULTI_SPEC  # one --input / --data-files pattern per chain
    + [
        ("name", "optional_str", None),
        ("truth", "optional_str", None),
        ("output", str, "extract.parquet"),
        ("cosmo-keys", list, ["Omega_c", "sigma8"]),
        ("field-statistic", bool, False),
        ("power-statistic", bool, False),
        ("ddof", int, 0),
    ]
    + _COMMON_SPEC,
    "born-rt": [("name", "optional_str", None)]
    + _LENSING_SPEC
    + _SOURCE_SPEC
    + _LENSING_POSTPROC_SPEC
    + _COMMON_SPEC,
    "dorian-rt": [("name", "optional_str", None)]
    + _LENSING_SPEC
    + _SOURCE_SPEC
    + _LENSING_POSTPROC_SPEC
    + [
        ("rt-interp", str, "bilinear"),
        ("no-parallel-transport", bool, False),
        ("with-born", bool, False),
    ],
}


def _to_param_key(flag: str) -> str:
    """Convert a CLI flag name to a params-dict key: 'Omega-b' -> 'omega_b'."""
    return flag.lower().replace("-", "_")


def _emit(parts: list[str], spec: list, params: dict) -> None:
    """Append CLI tokens for each entry in ``spec`` to ``parts``."""
    for flag, typ, _ in spec:
        key = _to_param_key(flag)
        value = params.get(key)

        if typ is bool:
            if value:
                parts.append(f"--{flag}")
        elif typ in ("optional_int", "optional_str", "optional_float"):
            if value is not None and value != "":
                parts.extend([f"--{flag}", str(value)])
        elif typ == "optional_list":
            if value:
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ is list:
            if value is not None:
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ in (int, float, str):
            if value is not None and value != "":
                parts.extend([f"--{flag}", str(value)])


def build_command(subcommand: str, params: dict) -> str:
    """Build one ``fli-launcher ... -- fli-<subcommand> ...`` command string."""
    if subcommand not in _SUBCOMMAND_SPECS:
        raise ValueError(f"Unknown subcommand: {subcommand!r}")

    if params.get("mode") == "command_only":
        parts: list[str] = [f"fli-{subcommand}"]
        if subcommand in _DISTRIBUTED_SUBCOMMANDS:
            _emit(parts, _DISTRIBUTED_SPEC, params)
        _emit(parts, _SUBCOMMAND_SPECS[subcommand], params)
        return " ".join(parts)

    parts: list[str] = ["fli-launcher"]
    _emit(parts, _SLURM_SPEC, params)

    parts += ["--", f"fli-{subcommand}"]

    _emit(parts, _SUBCOMMAND_SPECS[subcommand], params)

    return " ".join(parts)
