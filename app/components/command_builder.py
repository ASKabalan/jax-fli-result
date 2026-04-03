"""Build the CLI command string from collected parameters."""
from __future__ import annotations


# Per-subcommand argument specs: (flag, type, default)
# type is one of: str, int, float, bool, list, list_str, optional_int, optional_str
# For bool: flag is emitted only when value is True
# For list: --flag val1 val2 val3
# For optional_*: skipped when None

_SLURM_SPEC = [
    ("mode", str, "dryrun"),
    ("account", str, "XXX"),
    ("constraint", str, "h100"),
    ("gpus-per-node", int, 4),
    ("cpus-per-node", int, 16),
    ("tasks-per-node", "optional_int", None),
    ("nodes", int, 4),
    ("qos", str, "qos_gpu_h100-t3"),
    ("time-limit", str, "00:30:00"),
    ("slurm-script", "optional_str", None),
    ("pdim", list, [16, 1]),
    ("output-logs", str, "SLURM_LOGS"),
]

_SIM_SPEC = [
    ("lpt-order", int, 2),
    ("t0", float, 0.1),
    ("t1", float, 1.0),
    ("nb-steps", int, 30),
    ("interp", str, "none"),
    ("scheme", str, "bilinear"),
    ("paint-nside", "optional_int", None),
    ("kernel-width-arcmin", "optional_float", None),
    ("enable-x64", bool, False),
]

_COSMO_SPEC = [
    ("h", float, 0.6774),
    ("omega-b", float, 0.0486),
    ("omega-k", float, 0.0),
    ("w0", float, -1.0),
    ("wa", float, 0.0),
    ("n-s", float, 0.9667),
    ("omega-nu", float, 0.0),
]

_LENSING_SPEC = [
    ("nz-shear", "list_str", ["s3"]),
    ("min-z", float, 0.01),
    ("max-z", float, 1.5),
    ("n-integrate", int, 32),
]

_LIGHTCONE_SPEC = [
    ("nb-shells", int, 10),
    ("halo-fraction", int, 8),
    ("observer-position", list, [0.5, 0.5, 0.5]),
    ("ts", "optional_list", None),
    ("ts-near", "optional_list", None),
    ("ts-far", "optional_list", None),
    ("drift-on-lightcone", bool, False),
    ("min-width", float, 50.0),
]


def _to_param_key(flag: str) -> str:
    """Convert CLI flag name to Python parameter key: 'gpus-per-node' -> 'gpus_per_node'."""
    return flag.replace("-", "_")


def build_command(subcommand: str, params: dict) -> str:
    """Build a `fli-launcher <subcommand> ...` command string."""
    parts = ["fli-launcher", subcommand]

    specs = _get_specs_for(subcommand)
    for flag, typ, default in specs:
        key = _to_param_key(flag)
        value = params.get(key)

        if typ == bool:
            if value:
                parts.append(f"--{flag}")
        elif typ in ("optional_int", "optional_str", "optional_float", "optional_list"):
            # Skip None or empty string
            if value is not None and value != "":
                if typ == "optional_list":
                    parts.append(f"--{flag}")
                    parts.extend(str(v) for v in value)
                else:
                    parts.extend([f"--{flag}", str(value)])
        elif typ == list:
            if value is not None and value != default:
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ == "list_str":
            # list of strings (omega-c/sigma8/seed which can be range strings)
            if value:  # skip None and empty list
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ in (int, float, str):
            if value is not None and value != default:
                parts.extend([f"--{flag}", str(value)])

    return " ".join(parts)


def _get_specs_for(subcommand: str) -> list:
    """Return the full argument spec list for a subcommand."""
    specs = {
        "simulate": _SLURM_SPEC + _SIM_SPEC + _COSMO_SPEC + _LENSING_SPEC + _LIGHTCONE_SPEC + [
            ("output-dir", str, "results/cosmology_runs"),
            ("simulation-type", str, "nbody"),
            # Output target (mutually exclusive — only one emitted)
            ("nside", "optional_int", None),
            ("flatsky-npix", "optional_list", None),
            ("field-size", "optional_list", None),
            ("density", bool, False),
            ("mesh-size", list, [64, 64, 64, 32, 32, 32]),
            ("box-size", list, [1000.0, 1000.0, 1000.0]),
            ("omega-c", "list_str", []),
            ("sigma8", "list_str", []),
            ("seed", "list_str", []),
            ("shell-spacing", str, "comoving"),
            ("solver", str, "kdk"),
            ("density-widths", "optional_list", None),
            ("perf", bool, False),
            ("iterations", "optional_int", None),
        ],
        "grid": _SLURM_SPEC + _SIM_SPEC + _COSMO_SPEC + _LENSING_SPEC + _LIGHTCONE_SPEC + [
            ("output-dir", str, "results/grid_runs"),
            ("simulation-type", str, "nbody"),
            # Output target (mutually exclusive — only one emitted)
            ("nside", "optional_int", None),
            ("flatsky-npix", "optional_list", None),
            ("field-size", "optional_list", None),
            ("density", bool, False),
            ("mesh-size", list, [64, 64, 64, 32, 32, 32]),
            ("box-size", list, [500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0]),
            ("omega-c", "list_str", []),
            ("sigma8", "list_str", []),
            ("seed", "list_str", []),
            ("shell-spacing", str, "comoving"),
            ("solver", str, "kdk"),
            ("density-widths", "optional_list", None),
        ],
        "samples": _SLURM_SPEC + _SIM_SPEC + _LENSING_SPEC + _LIGHTCONE_SPEC + [
            ("output-dir", str, "test_fli_samples"),
            ("model", str, "mock"),
            ("mesh-size", list, [64, 64, 64]),
            ("box-size", list, [250.0, 250.0, 250.0]),
            ("nside", int, 64),
            ("num-samples", int, 10),
            ("chains", list, [0, 1, 2, 3]),
            ("batches", list, [0, 1, 2, 3, 4, 5]),
            ("equal-vol", bool, False),
        ],
        "infer": _SLURM_SPEC + _SIM_SPEC + _LENSING_SPEC + _LIGHTCONE_SPEC + [
            ("observable-dir", str, "observables"),
            ("observable", str, ""),
            ("output-dir", str, "results/inference_runs"),
            ("mesh-size", list, [16, 16, 16]),
            ("box-size", list, [1000.0, 1000.0, 1000.0]),
            ("chain-index", int, 0),
            ("adjoint", str, "checkpointed"),
            ("checkpoints", int, 10),
            ("num-warmup", int, 1),
            ("num-samples", int, 1),
            ("batch-count", int, 2),
            ("sampler", str, "NUTS"),
            ("backend", str, "blackjax"),
            ("sigma-e", float, 0.26),
            ("sample", list, ["cosmo", "ic"]),
            ("initial-condition", "optional_str", None),
            ("init-cosmo", bool, False),
            ("equal-vol", bool, False),
            ("omega-c", float, 0.2589),
            ("sigma8", float, 0.8159),
            ("h", float, 0.6774),
            ("seed", int, 0),
        ],
        "extract": _SLURM_SPEC + [
            ("input-dir", str, "test_fli_samples"),
            ("repo-id", "optional_str", None),
            ("config", "optional_list", None),
            ("truth-parquet", str, "test_fli_samples/chain_0/samples/samples_0.parquet"),
            ("output-file", str, "results/extracts/extract.parquet"),
            ("set-name", str, "my_extract"),
            ("cosmo-keys", list, ["Omega_c", "sigma8"]),
            ("field-statistic", bool, True),
            ("power-statistic", bool, True),
            ("ddof", int, 0),
            ("enable-x64", bool, False),
        ],
        "born-rt": _SLURM_SPEC + _LENSING_SPEC + [
            ("input-dir", str, "results/cosmology_runs"),
            ("output-dir", str, "results/lensing/multi_shell"),
            ("enable-x64", bool, False),
        ],
        "dorian-rt": _SLURM_SPEC + _LENSING_SPEC + [
            ("input-dir", str, "results/cosmology_runs"),
            ("output-dir", str, "results/lensing/multi_shell_raytrace"),
            ("rt-interp", str, "bilinear"),
            ("no-parallel-transport", bool, False),
        ],
    }
    return specs.get(subcommand, [])
