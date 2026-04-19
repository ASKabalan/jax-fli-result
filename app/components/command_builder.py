"""Build the CLI command string from collected parameters.

Emits:

    fli-launcher <SLURM flags> -- fli-<subcommand> <script flags>

The SLURM spec is shared across all subcommands (mirrors
``jax_fli.scripts.parser.add_slurm_args``). Each subcommand has its own
script-side spec below. ``build_command`` always returns a single string —
gridding/sweeping is not supported (the launcher runs one job per call).
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

# Science-side spec fragments reused across several subcommands.

_INTEGRATION_SPEC = [
    ("sim-mode", str, "lensing"),
    ("lpt-order", int, 2),
    ("t0", float, 0.001),
    ("t1", float, 1.0),
    ("nb-steps", int, 30),
    ("nb-shells", int, 10),
    ("interp", str, "none"),
    ("solver", str, "kdk"),
    ("time-stepping", str, "a"),
    ("shell-spacing", str, "comoving"),
    ("dealiased", bool, False),
    ("exact-growth", bool, False),
    ("gradient-order", int, 1),
    ("laplace-fd", bool, False),
    ("density-widths", "optional_list", None),
    ("ts", "optional_list", None),
    ("ts-near", "optional_list", None),
    ("ts-far", "optional_list", None),
    ("drift-on-lightcone", bool, False),
    ("min-width", float, 50.0),
]

_LENSING_SPEC = [
    ("nz-shear", list, ["s3"]),
    ("min-z", float, 0.01),
    ("max-z", float, 1.5),
    ("n-integrate", int, 32),
]

_SIM_SETTINGS_SPEC = [
    ("mesh-size", list, [64, 64, 64]),
    ("box-size", list, [250.0, 250.0, 250.0]),
    ("halo-multiplier", float, 0.5),
    ("observer-position", list, [0.5, 0.5, 0.5]),
    ("seed", int, 0),
    ("scheme", str, "bilinear"),
    ("paint-nside", "optional_int", None),
    ("kernel-width-arcmin", "optional_float", None),
    ("enable-x64", bool, False),
    # Output target (mutually exclusive — only one should be set per call).
    ("nside", "optional_int", None),
    ("flatsky-npix", "optional_list", None),
    ("field-size", "optional_list", None),
    ("density", bool, False),
]

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

_PRIOR_SPEC = [
    ("sample", list, ["cosmo", "ic"]),
    ("prior-omega-c", list, [0.1, 0.5]),
    ("prior-sigma8", list, [0.6, 1.0]),
    ("prior-h", list, [0.5, 0.9]),
    ("prior-ic-gaussian", list, [0.0, 1.0]),
]

_INFER_SPEC = [
    ("initial-condition", "optional_str", None),
    ("init-cosmo", bool, False),
    ("sigma-e", float, 0.26),
    ("num-warmup", int, 500),
    ("num-samples", int, 1000),
    ("batch-count", int, 5),
    ("adjoint", str, "checkpointed"),
    ("checkpoints", int, 10),
    ("sampler", str, "NUTS"),
    ("backend", str, "numpyro"),
    ("no-progress-bar", bool, False),
]

# Per-subcommand script specs (what goes AFTER the `--`).
_SUBCOMMAND_SPECS: dict[str, list] = {
    "simulate": _SIM_SETTINGS_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _COSMO_SPEC
    + [
        ("output", str, "sim_output.parquet"),
        ("name", "optional_str", None),
        ("perf", bool, False),
        ("iterations", int, 5),
    ],
    "samples": _SIM_SETTINGS_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _PRIOR_SPEC
    + [
        ("path", str, "test_fli_samples"),
        ("model", str, "full"),
        ("sigma-e", float, 0.26),
        ("num-samples", int, 100),
        ("batch-id", int, 0),
        ("initial-condition", "optional_str", None),
    ],
    "infer": _SIM_SETTINGS_SPEC
    + _INTEGRATION_SPEC
    + _LENSING_SPEC
    + _PRIOR_SPEC
    + _INFER_SPEC
    + [
        ("observable", str, ""),
        ("path", str, "results/inference_runs"),
    ],
    "2pcf": _LENSING_SPEC
    + _PRIOR_SPEC
    + [
        ("observable", str, ""),
        ("path", str, "results/2pcf_inference"),
        ("nside", "optional_int", None),
        ("flatsky-npix", "optional_list", None),
        ("field-size", "optional_list", None),
        ("lmax", int, 2047),
        ("f-sky", float, 1.0),
        ("sigma-e", float, 0.26),
        ("nonlinear-fn", str, "halofit"),
        ("chain-index", int, 0),
        ("num-warmup", int, 100),
        ("num-samples", int, 500),
        ("batch-count", int, 10),
        ("sampler", str, "NUTS"),
        ("backend", str, "numpyro"),
        ("seed", int, 0),
        ("no-progress-bar", bool, False),
        ("enable-x64", bool, False),
    ],
    "extract": [
        ("path", "optional_str", None),
        ("repo-id", "optional_str", None),
        ("config", "optional_list", None),
        ("set-name", "optional_str", None),
        ("truth", "optional_str", None),
        ("output", str, "extract.parquet"),
        ("cosmo-keys", list, ["Omega_c", "sigma8"]),
        ("field-statistic", bool, False),
        ("power-statistic", bool, False),
        ("ddof", int, 0),
        ("enable-x64", bool, False),
    ],
    "born-rt": _LENSING_SPEC
    + [
        ("input", str, "results/cosmology_runs"),
        ("output", str, "results/lensing/multi_shell"),
        ("enable-x64", bool, False),
    ],
    "dorian-rt": _LENSING_SPEC
    + [
        ("input", str, "results/cosmology_runs"),
        ("output", str, "results/lensing/multi_shell_raytrace"),
        ("rt-interp", str, "bilinear"),
        ("no-parallel-transport", bool, False),
    ],
    "spectra": [
        ("folder", str, "results"),
        ("regex", str, r".*\.parquet$"),
        ("recursive", bool, False),
        ("force-regen", bool, False),
        ("normalization", str, "global"),
        ("ell-edges", "optional_list", None),
        ("lmax", "optional_int", None),
        ("kmax", "optional_float", None),
        ("method", str, "healpy"),
        ("kedges", "optional_list", None),
        ("multipoles", list, [0]),
        ("los", list, [0.0, 0.0, 1.0]),
        ("batch-size", "optional_int", None),
        ("enable-x64", bool, False),
    ],
}


def _to_param_key(flag: str) -> str:
    """Convert a CLI flag name to a params-dict key: 'Omega-b' -> 'omega_b'."""
    return flag.lower().replace("-", "_")


def _emit(
    parts: list[str], spec: list, params: dict, positional_first: str | None = None
) -> None:
    """Append CLI tokens for each entry in ``spec`` to ``parts``.

    If ``positional_first`` is set, the matching spec entry is emitted as a
    positional argument (no ``--flag`` prefix) at the front. Used by
    ``fli-spectra`` where ``folder`` is positional.
    """
    positional_value = None
    for flag, typ, _ in spec:
        key = _to_param_key(flag)
        value = params.get(key)

        if flag == positional_first:
            if value is not None:
                positional_value = str(value)
            continue

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

    if positional_first is not None and positional_value is not None:
        parts.append(positional_value)


def build_command(subcommand: str, params: dict) -> str:
    """Build one ``fli-launcher ... -- fli-<subcommand> ...`` command string."""
    if subcommand not in _SUBCOMMAND_SPECS:
        raise ValueError(f"Unknown subcommand: {subcommand!r}")

    parts: list[str] = ["fli-launcher"]
    _emit(parts, _SLURM_SPEC, params)

    parts += ["--", f"fli-{subcommand}"]

    # fli-spectra takes `folder` positionally; everything else takes --flag VALUE.
    positional_first = "folder" if subcommand == "spectra" else None
    _emit(
        parts, _SUBCOMMAND_SPECS[subcommand], params, positional_first=positional_first
    )

    return " ".join(parts)
