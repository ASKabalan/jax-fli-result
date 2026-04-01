"""grid subcommand — wraps simulation_grid.bash / fli-grid."""
from __future__ import annotations

import os

from launcher.parser import (
    add_common_sim_args,
    add_lensing_args,
    add_lightcone_args,
    add_slurm_args,
    dispatch,
)

# Defaults matching simulation_grid.bash
_DEFAULT_TS_NEAR = [
    "0.3938", "0.4052", "0.4165", "0.4276", "0.4387",
    "0.4497", "0.4606", "0.4714", "0.4822", "0.4929",
]
_DEFAULT_TS_FAR = [
    "0.4052", "0.4165", "0.4276", "0.4387", "0.4497",
    "0.4606", "0.4714", "0.4822", "0.4929", "0.5036",
]


def add_subparser(sub):
    p = sub.add_parser(
        "grid",
        help="Submit a single fli-grid job (full parameter-grid exploration in one process)",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_lensing_args(p)
    add_lightcone_args(p)

    g = p.add_argument_group("grid")
    g.add_argument("--output-dir", default="results/grid_runs")
    g.add_argument(
        "--simulation-type",
        choices=["lpt", "nbody", "lensing"],
        default="nbody",
    )
    g.add_argument(
        "--mesh-size",
        nargs="+",
        type=int,
        default=[64, 64, 64, 32, 32, 32],
        help="Flat list grouped into triples, e.g. 64 64 64 128 128 128",
    )
    g.add_argument(
        "--box-size",
        nargs="+",
        type=float,
        default=[500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0],
        help="Flat list grouped into triples",
    )
    g.add_argument(
        "--omega-c",
        nargs="+",
        default=["0.2"],
        help="Values or range strings, e.g. 0.2 or '0.2:0.4:0.05'",
    )
    g.add_argument(
        "--sigma8",
        nargs="+",
        default=["0.8"],
        help="Values or range strings",
    )
    g.add_argument("--seed", nargs="+", default=["0"])
    g.add_argument("--nside", nargs="+", type=int, default=[512])
    g.add_argument(
        "--shell-spacing",
        choices=["comoving", "equal_vol", "a", "growth"],
        default="comoving",
    )
    g.add_argument("--solver", choices=["kdk", "dkd", "bf"], default="kdk")
    g.add_argument(
        "--density-widths",
        nargs="*",
        default=None,
        help="Space-separated scalar density widths",
    )

    p.set_defaults(
        time_limit="24:00:00",   # grid runs ALL combos — set generously
        drift_on_lightcone=True,
        nb_shells=None,          # fli-grid handles its own shell defaults
        ts_near=_DEFAULT_TS_NEAR,
        ts_far=_DEFAULT_TS_FAR,
        func=run,
    )


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    px, py = args.pdim
    job_name = f"fli_grid_{args.simulation_type}"

    fli_cmd = [
        "fli-grid", args.simulation_type,
        "--mesh-size", *[str(m) for m in args.mesh_size],
        "--box-size", *[str(b) for b in args.box_size],
        "--Omega-c", *[str(v) for v in args.omega_c],
        "--sigma8", *[str(v) for v in args.sigma8],
        "--seed", *[str(v) for v in args.seed],
    ]
    if args.nb_shells is not None:
        fli_cmd += ["--nb-shells", str(args.nb_shells)]
    if args.ts:
        fli_cmd += ["--ts", *[str(t) for t in args.ts]]
    if args.ts_near:
        fli_cmd += ["--ts-near", *[str(t) for t in args.ts_near]]
    if args.ts_far:
        fli_cmd += ["--ts-far", *[str(t) for t in args.ts_far]]
    fli_cmd += [
        "--nb-steps", str(args.nb_steps),
        "--nside", *[str(n) for n in args.nside],
        "--t0", str(args.t0),
        "--t1", str(args.t1),
        "--lpt-order", str(args.lpt_order),
        "--halo-fraction", str(args.halo_fraction),
        "--pdim", str(px), str(py),
        "--nodes", str(args.nodes),
        "--interp", args.interp,
        "--scheme", args.scheme,
    ]
    if args.paint_nside is not None:
        fli_cmd += ["--paint-nside", str(args.paint_nside)]
    if args.drift_on_lightcone:
        fli_cmd.append("--drift-on-lightcone")
    fli_cmd += ["--min-width", str(args.min_width)]
    fli_cmd += ["--shell-spacing", args.shell_spacing]
    fli_cmd += ["--solver", args.solver]
    if args.density_widths:
        fli_cmd += ["--density-widths", *[str(d) for d in args.density_widths]]
    if args.simulation_type == "lensing":
        fli_cmd += [
            "--nz-shear", *[str(v) for v in args.nz_shear],
            "--min-z", str(args.min_z),
            "--max-z", str(args.max_z),
            "--n-integrate", str(args.n_integrate),
        ]
    fli_cmd += ["--h", "0.6774", "--output-dir", args.output_dir]
    if args.enable_x64:
        fli_cmd.append("--enable-x64")

    dispatch(args, job_name, "FLI_GRID", fli_cmd)
