"""Build the CLI command string from collected parameters."""
from __future__ import annotations

DEFAULT_NAME_TEMPLATE: str = "%constraint%_cosmo_M%mesh_size%_B%box_size%_STEPS%nb_steps%_c%omega_c%_S8%sigma8%_s%seed%"
try:
    from jax_fli.scripts.parser import DEFAULT_NAME_TEMPLATE  # noqa: F811
except ImportError:
    pass

# Per-subcommand argument specs: (flag, type, default)
# type is one of: str, int, float, bool, list, list_str, optional_int, optional_str
# For bool: flag is emitted only when value is True
# For list: --flag val1 val2 val3
# For optional_*: skipped when None

_SLURM_SPEC = [
    ("mode", str, "dryrun"),
    ("account", str, "XXX"),
    ("constraint", "optional_str", None),
    ("gpus-per-node", int, 4),
    ("cpus-per-node", int, 16),
    ("tasks-per-node", "optional_int", None),
    ("nodes", int, 4),
    ("qos", str, "qos_gpu_h100-t3"),
    ("time-limit", str, "00:30:00"),
    ("slurm-script", "optional_str", None),
    ("pdim", "optional_list", None),
    ("output-logs", str, "SLURM_LOGS"),
]

_SIM_SPEC = [
    ("lpt-order", int, 2),
    ("t0", float, 0.001),
    ("t1", float, 1.0),
    ("nb-steps", int, 30),
    ("interp", str, "none"),
    ("scheme", str, "bilinear"),
    ("paint-nside", "optional_int", None),
    ("kernel-width-arcmin", "optional_float", None),
    ("enable-x64", bool, False),
    ("dealiased", bool, False),
    ("exact-growth", bool, False),
    ("gradient-order", int, 1),
    ("laplace-fd", bool, False),
    ("time-stepping", str, "a"),
]

_COSMO_SPEC = [
    ("h", float, 0.6774),
    ("Omega-b", float, 0.0486),
    ("Omega-k", float, 0.0),
    ("w0", float, -1.0),
    ("wa", float, 0.0),
    ("n-s", float, 0.9667),
    ("Omega-nu", float, 0.0),
]

_LENSING_SPEC = [
    ("nz-shear", "list_str", ["s3"]),
    ("min-z", float, 0.01),
    ("max-z", float, 1.5),
    ("n-integrate", int, 32),
]

_LIGHTCONE_SPEC = [
    ("nb-shells", int, 10),
    ("halo-multiplier", float, 0.5),
    ("observer-position", list, [0.5, 0.5, 0.5]),
    ("ts", "optional_list", None),
    ("ts-near", "optional_list", None),
    ("ts-far", "optional_list", None),
    ("drift-on-lightcone", bool, False),
    ("min-width", float, 50.0),
]


def _to_param_key(flag: str) -> str:
    """Convert CLI flag name to Python parameter key: 'Omega-b' -> 'omega_b'.

    Lowercases the flag name so that both 'Omega-b' and 'omega-b' map to
    the same dict key 'omega_b', keeping the GUI param dicts stable.
    """
    return flag.lower().replace("-", "_")


def build_command(subcommand: str, params: dict) -> str:
    """Build a `fli-launcher <subcommand> ...` command string.

    Parameters
    ----------
    subcommand : str
        The subcommand for which to build the command.
    params : dict
        A dictionary of parameter values.
    """
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
            if value is not None:
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ == "list_str":
            # list of strings (omega-c/sigma8/seed which can be range strings)
            if value:  # skip None and empty list
                parts.append(f"--{flag}")
                parts.extend(str(v) for v in value)
        elif typ in (int, float, str):
            if value is not None:
                parts.extend([f"--{flag}", str(value)])

    return " ".join(parts)


def _get_specs_for(subcommand: str) -> list:
    """Return the full argument spec list for a subcommand."""
    specs = {
        "simulate": _SLURM_SPEC
        + _SIM_SPEC
        + _COSMO_SPEC
        + _LENSING_SPEC
        + _LIGHTCONE_SPEC
        + [
            ("output-dir", str, "results/cosmology_runs"),
            ("name-template", str, DEFAULT_NAME_TEMPLATE),
            ("simulation-type", str, "nbody"),
            # Output target (mutually exclusive — only one emitted)
            ("nside", "optional_int", None),
            ("flatsky-npix", "optional_list", None),
            ("field-size", "optional_list", None),
            ("density", bool, False),
            ("mesh-size", list, [64, 64, 64, 32, 32, 32]),
            ("box-size", list, [1000.0, 1000.0, 1000.0]),
            ("Omega-c", "list_str", []),
            ("sigma8", "list_str", []),
            ("seed", "list_str", []),
            ("shell-spacing", str, "comoving"),
            ("solver", str, "kdk"),
            ("density-widths", "optional_list", None),
            ("perf", bool, False),
            ("iterations", "optional_int", None),
        ],
        "samples": _SLURM_SPEC
        + _SIM_SPEC
        + _LENSING_SPEC
        + _LIGHTCONE_SPEC
        + [
            ("output-dir", str, "test_fli_samples"),
            ("model", str, "mock"),
            ("mesh-size", list, [64, 64, 64]),
            ("box-size", list, [250.0, 250.0, 250.0]),
            ("nside", int, 64),
            ("num-samples", int, 10),
            ("chains", list, [0, 1, 2, 3]),
            ("batches", list, [0, 1, 2, 3, 4, 5]),
            ("equal-vol", bool, False),
            ("sample", "list_str", ["cosmo", "ic"]),
            ("prior-omega-c", list, [0.1, 0.5]),
            ("prior-sigma8", list, [0.6, 1.0]),
            ("prior-h", list, [0.5, 0.9]),
            ("prior-ic-gaussian", list, [0.0, 1.0]),
            ("initial-condition", "optional_str", None),
        ],
        "infer": _SLURM_SPEC
        + _SIM_SPEC
        + _LENSING_SPEC
        + _LIGHTCONE_SPEC
        + [
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
            ("sample", "list_str", ["cosmo", "ic"]),
            ("prior-omega-c", list, [0.1, 0.5]),
            ("prior-sigma8", list, [0.6, 1.0]),
            ("prior-h", list, [0.5, 0.9]),
            ("initial-condition", "optional_str", None),
            ("init-cosmo", bool, False),
            ("equal-vol", bool, False),
            ("seed", int, 0),
        ],
        "2pcf": _SLURM_SPEC
        + _LENSING_SPEC
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
            ("backend", str, "blackjax"),
            ("sample", "list_str", ["cosmo"]),
            ("prior-omega-c", list, [0.1, 0.5]),
            ("prior-sigma8", list, [0.6, 1.0]),
            ("prior-h", list, [0.5, 0.9]),
            ("seed", int, 0),
            ("enable-x64", bool, False),
        ],
        "spectra": _SLURM_SPEC
        + [
            ("folder", str, "results"),
            ("regex", str, r".*\.parquet$"),
            ("recursive", bool, False),
            ("force-regen", bool, False),
            ("normalization", str, "global"),
            ("ell-edges", "optional_list", None),
            ("lmax", "optional_int", None),
            ("method", str, "healpy"),
            ("kedges", "optional_list", None),
            ("multipoles", list, [0]),
            ("los", list, [0.0, 0.0, 1.0]),
            ("batch-size", "optional_int", None),
            ("enable-x64", bool, False),
        ],
        "extract": _SLURM_SPEC
        + [
            ("input-dir", str, "test_fli_samples"),
            ("repo-id", "optional_str", None),
            ("config", "optional_list", None),
            (
                "truth-parquet",
                str,
                "test_fli_samples/chain_0/samples/samples_0.parquet",
            ),
            ("output-file", str, "results/extracts/extract.parquet"),
            ("set-name", str, "my_extract"),
            ("cosmo-keys", list, ["Omega_c", "sigma8"]),
            ("field-statistic", bool, True),
            ("power-statistic", bool, True),
            ("ddof", int, 0),
            ("enable-x64", bool, False),
        ],
        "born-rt": _SLURM_SPEC
        + _LENSING_SPEC
        + [
            ("input-dir", str, "results/cosmology_runs"),
            ("output-dir", str, "results/lensing/multi_shell"),
            ("enable-x64", bool, False),
        ],
        "dorian-rt": _SLURM_SPEC
        + _LENSING_SPEC
        + [
            ("input-dir", str, "results/cosmology_runs"),
            ("output-dir", str, "results/lensing/multi_shell_raytrace"),
            ("rt-interp", str, "bilinear"),
            ("no-parallel-transport", bool, False),
        ],
    }
    return specs.get(subcommand, [])
