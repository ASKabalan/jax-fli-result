"""simulate subcommand — wraps simulation_script.bash / fli-simulate."""
from __future__ import annotations

import os

from launcher.parser import (
    add_common_sim_args,
    add_cosmo_args,
    add_lensing_args,
    add_lightcone_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "simulate",
        help="Submit fli-simulate jobs over a mesh × box × cosmology × seed grid",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_cosmo_args(p)
    add_lensing_args(p)
    add_lightcone_args(p)

    g = p.add_argument_group("simulate")
    g.add_argument("--output-dir", default="results/cosmology_runs")
    g.add_argument(
        "--simulation-type",
        choices=["lpt", "nbody", "lensing"],
        default="nbody",
    )
    g.add_argument("--nside", type=int, default=64)
    g.add_argument(
        "--mesh-size",
        nargs="+",
        type=int,
        default=[64, 64, 64, 32, 32, 32],
        help="Flat list of mesh sizes grouped into triples, e.g. 64 64 64 32 32 32",
    )
    g.add_argument(
        "--box-size",
        nargs="+",
        type=float,
        default=[1000.0, 1000.0, 1000.0],
        help="Flat list of box sizes grouped into triples, e.g. 1000 1000 1000",
    )
    g.add_argument("--omega-c", nargs="+", type=float, default=[0.2589])
    g.add_argument("--sigma8", nargs="+", type=float, default=[0.8159])
    g.add_argument("--seed", nargs="+", type=int, default=[0])
    g.add_argument(
        "--shell-spacing",
        choices=["comoving", "equal_vol", "a", "growth"],
        default="comoving",
    )
    g.add_argument("--solver", choices=["kdk", "dkd", "bf"], default="kdk")

    # drift-on-lightcone is ON by default in the bash script
    p.set_defaults(drift_on_lightcone=True, func=run)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    mesh_list = args.mesh_size
    if len(mesh_list) % 3 != 0:
        raise ValueError(f"--mesh-size must be a multiple of 3 values, got {len(mesh_list)}")
    meshes = [mesh_list[i : i + 3] for i in range(0, len(mesh_list), 3)]

    box_list = args.box_size
    if len(box_list) % 3 != 0:
        raise ValueError(f"--box-size must be a multiple of 3 values, got {len(box_list)}")
    boxes = [box_list[i : i + 3] for i in range(0, len(box_list), 3)]

    px, py = args.pdim

    print("Launching cosmology simulations...")

    for box in boxes:
        # Replicate bash: join with "x", strip ".0" suffixes
        box_name = "x".join(str(b) for b in box).replace(".0", "")

        for mesh in meshes:
            mesh_name = "x".join(str(m) for m in mesh)

            for oc in args.omega_c:
                for s8 in args.sigma8:
                    for sd in args.seed:
                        job_name = (
                            f"{args.constraint}_cosmo_M{mesh_name}_B{box_name}"
                            f"_STEPS{args.nb_steps}_c{oc}_S8{s8}_s{sd}"
                        )
                        out_file = f"{args.output_dir}/{job_name}.parquet"

                        fli_cmd = [
                            "fli-simulate", args.simulation_type,
                            "--mesh-size", *[str(m) for m in mesh],
                            "--box-size", *[str(b) for b in box],
                            "--pdim", str(px), str(py),
                            "--nodes", str(args.nodes),
                            "--halo-fraction", str(args.halo_fraction),
                            "--observer-position", *[str(v) for v in args.observer_position],
                            "--nside", str(args.nside),
                        ]
                        if not args.ts and not args.ts_near and not args.ts_far:
                            fli_cmd += ["--nb-shells", str(args.nb_shells)]
                        if args.ts:
                            fli_cmd += ["--ts", *[str(t) for t in args.ts]]
                        if args.ts_near:
                            fli_cmd += ["--ts-near", *[str(t) for t in args.ts_near]]
                        if args.ts_far:
                            fli_cmd += ["--ts-far", *[str(t) for t in args.ts_far]]
                        fli_cmd += [
                            "--t0", str(args.t0),
                            "--nb-steps", str(args.nb_steps),
                            "--t1", str(args.t1),
                            "--lpt-order", str(args.lpt_order),
                            "--interp", args.interp,
                            "--scheme", args.scheme,
                        ]
                        if args.paint_nside is not None:
                            fli_cmd += ["--paint-nside", str(args.paint_nside)]
                        if args.kernel_width_arcmin is not None:
                            fli_cmd += ["--kernel-width-arcmin", str(args.kernel_width_arcmin)]
                        if args.drift_on_lightcone:
                            fli_cmd.append("--drift-on-lightcone")
                        fli_cmd += [
                            "--min-width", str(args.min_width),
                            "--shell-spacing", args.shell_spacing,
                            "--solver", args.solver,
                        ]
                        if args.simulation_type == "lensing":
                            fli_cmd += [
                                "--nz-shear", *[str(v) for v in args.nz_shear],
                                "--min-z", str(args.min_z),
                                "--max-z", str(args.max_z),
                                "--n-integrate", str(args.n_integrate),
                            ]
                        fli_cmd += [
                            "--Omega-c", str(oc),
                            "--sigma8", str(s8),
                            "--Omega-b", str(args.omega_b),
                            "--h", str(args.h),
                            "--n-s", str(args.n_s),
                            "--Omega-k", str(args.omega_k),
                            "--w0", str(args.w0),
                            "--wa", str(args.wa),
                            "--Omega-nu", str(args.omega_nu),
                            "--seed", str(sd),
                            "--output", out_file,
                            "--perf",
                            "--iterations", "3",
                        ]
                        if args.enable_x64:
                            fli_cmd.append("--enable-x64")

                        dispatch(args, job_name, "FLI_SIMULATION", fli_cmd)
