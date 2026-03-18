"""samples subcommand — wraps generate_samples.bash / fli-samples."""
from __future__ import annotations

import os

from launcher.parser import (
    add_common_sim_args,
    add_lensing_args,
    add_lightcone_args,
    add_slurm_args,
    dispatch,
)


def add_subparser(sub):
    p = sub.add_parser(
        "samples",
        help="Submit fli-samples jobs across chains × batches",
    )
    add_slurm_args(p)
    add_common_sim_args(p)
    add_lensing_args(p)
    add_lightcone_args(p)

    g = p.add_argument_group("samples")
    g.add_argument("--output-dir", default="test_fli_samples")
    g.add_argument("--model", choices=["full", "mock"], default="mock")
    g.add_argument("--mesh-size", nargs=3, type=int, default=[64, 64, 64],
                   metavar=("MX", "MY", "MZ"))
    g.add_argument("--box-size", nargs=3, type=float, default=[250.0, 250.0, 250.0],
                   metavar=("BX", "BY", "BZ"))
    g.add_argument("--nside", type=int, default=64)
    g.add_argument("--num-samples", type=int, default=10)
    g.add_argument("--chains", nargs="+", type=int, default=[0, 1, 2, 3],
                   help="Chain IDs to run")
    g.add_argument("--batches", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                   help="Batch IDs to run")

    # generate_samples.bash uses t0=0.01, nb_steps=100, nb_shells=8
    p.set_defaults(t0=0.01, nb_steps=100, nb_shells=8, func=run)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    px, py = args.pdim

    for chain in args.chains:
        for batch in args.batches:
            job_name = f"{args.constraint}_samples_chain{chain}_batch{batch}"
            print(f"Submitting {job_name}")

            fli_cmd = [
                "fli-samples",
                "--model", args.model,
                "--mesh-size", *[str(m) for m in args.mesh_size],
                "--box-size", *[str(b) for b in args.box_size],
                "--pdim", str(px), str(py),
                "--nodes", str(args.nodes),
                "--nside", str(args.nside),
                "--lpt-order", str(args.lpt_order),
                "--t0", str(args.t0),
                "--t1", str(args.t1),
                "--nb-steps", str(args.nb_steps),
                "--nb-shells", str(args.nb_shells),
                "--halo-fraction", str(args.halo_fraction),
                "--observer-position", *[str(v) for v in args.observer_position],
                "--nz-shear", args.nz_shear,
                "--min-z", str(args.min_z),
                "--max-z", str(args.max_z),
                "--n-integrate", str(args.n_integrate),
                "--interp", args.interp,
                "--scheme", args.scheme,
            ]
            if args.paint_nside is not None:
                fli_cmd += ["--paint-nside", str(args.paint_nside)]
            fli_cmd += [
                "--num-samples", str(args.num_samples),
                "--seed", str(batch),
                "--path", f"{args.output_dir}/chain_{chain}",
                "--batch-id", str(batch),
            ]
            if args.enable_x64:
                fli_cmd.append("--enable-x64")

            dispatch(args, job_name, "FLI_SAMPLES", fli_cmd)
