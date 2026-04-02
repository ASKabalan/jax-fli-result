"""Shared argument-group builders and dispatch logic for all subcommands."""
from __future__ import annotations

import os
import subprocess
import sys


def add_slurm_args(p):
    """Cluster/SLURM arguments common to all subcommands."""
    g = p.add_argument_group("SLURM / cluster")
    g.add_argument("--mode", choices=["local", "sbatch", "dryrun"], default="dryrun",
                   help="Execution mode (default: dryrun)")
    g.add_argument("--account", default="XXX")
    g.add_argument("--constraint", default="h100")
    g.add_argument("--gpus-per-node", type=int, default=4)
    g.add_argument("--cpus-per-node", type=int, default=16)
    g.add_argument("--tasks-per-node", type=int, default=None,
                   help="Defaults to --gpus-per-node when not set")
    g.add_argument("--nodes", type=int, default=4)
    g.add_argument("--qos", default="qos_gpu_h100-t3")
    g.add_argument("--time-limit", default="00:30:00")
    g.add_argument("--slurm-script", default=None,
                   help="Path to SLURM job script (required when --mode=sbatch)")
    g.add_argument("--pdim", nargs=2, type=int, default=[16, 1], metavar=("PX", "PY"))
    g.add_argument("--output-logs", default="SLURM_LOGS",
                   help="Directory for SLURM log files")


def add_common_sim_args(p):
    """Simulation parameters shared across simulate/samples/infer/grid subcommands."""
    g = p.add_argument_group("simulation")
    g.add_argument("--lpt-order", type=int, default=2)
    g.add_argument("--t0", type=float, default=0.1)
    g.add_argument("--t1", type=float, default=1.0)
    g.add_argument("--nb-steps", type=int, default=30)
    g.add_argument("--interp", default="none")
    g.add_argument("--scheme", default="bilinear",
                   choices=["ngp", "bilinear", "rbf_neighbor"])
    g.add_argument("--paint-nside", type=int, default=None)
    g.add_argument("--kernel-width-arcmin", type=float, default=None, dest="kernel_width_arcmin")
    g.add_argument("--enable-x64", action="store_true")


def add_cosmo_args(p):
    """Full cosmological parameters (beyond Omega_c / sigma8)."""
    g = p.add_argument_group("cosmology")
    g.add_argument("--h", type=float, default=0.6774, help="Hubble parameter (default: 0.6774)")
    g.add_argument("--omega-b", type=float, default=0.0486, dest="omega_b")
    g.add_argument("--omega-k", type=float, default=0.0, dest="omega_k")
    g.add_argument("--w0", type=float, default=-1.0)
    g.add_argument("--wa", type=float, default=0.0)
    g.add_argument("--n-s", type=float, default=0.9667, dest="n_s")
    g.add_argument("--omega-nu", type=float, default=0.0, dest="omega_nu")


def add_lensing_args(p):
    """Lensing parameters."""
    g = p.add_argument_group("lensing")
    g.add_argument("--nz-shear", nargs="+", default=["s3"])
    g.add_argument("--min-z", type=float, default=0.01)
    g.add_argument("--max-z", type=float, default=1.5)
    g.add_argument("--n-integrate", type=int, default=32)


def add_lightcone_args(p):
    """Lightcone / shell parameters."""
    g = p.add_argument_group("lightcone")
    g.add_argument("--nb-shells", type=int, default=10)
    g.add_argument("--halo-fraction", type=int, default=8)
    g.add_argument("--observer-position", nargs=3, type=float, default=[0.5, 0.5, 0.5],
                   metavar=("OX", "OY", "OZ"))
    g.add_argument("--ts", nargs="*", default=None,
                   help="Explicit shell-centre scale factors (overrides --nb-shells)")
    g.add_argument("--ts-near", nargs="*", default=None,
                   help="Near-side shell boundary scale factors")
    g.add_argument("--ts-far", nargs="*", default=None,
                   help="Far-side shell boundary scale factors")
    g.add_argument("--drift-on-lightcone", action="store_true")
    g.add_argument("--min-width", type=float, default=50.0)
    g.add_argument("--equal-vol", action="store_true",
                   help="Use equal-volume shell partitioning (fli-infer/fli-samples)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slurm_params(args):
    """Return (tasks_per_node, cpus_per_task, total_gpus)."""
    tpn = args.tasks_per_node if args.tasks_per_node is not None else args.gpus_per_node
    cpt = args.cpus_per_node // tpn
    total_gpus = args.gpus_per_node * args.nodes
    return tpn, cpt, total_gpus


def _print_dryrun(job_name, args, fli_cmd):
    tpn, cpt, _ = _slurm_params(args)
    print("=======================================================")
    print(f"Submitting job {job_name}")
    print("=======================================================")
    print(f"{'ACCOUNT':<16} | {args.account}")
    print(f"{'CONSTRAINT':<16} | {args.constraint}")
    print(f"{'TIME_LIMIT':<16} | {args.time_limit}")
    print(f"{'GPUS_PER_NODE':<16} | {args.gpus_per_node}")
    print(f"{'CPUS_PER_TASK':<16} | {cpt}")
    print(f"{'NODES':<16} | {args.nodes}")
    print(f"{'TASKS_PER_NODE':<16} | {tpn}")
    print(f"{'QOS':<16} | {args.qos}")
    print("*******************************************************")
    print(" ".join(fli_cmd))
    print("*******************************************************")
    print("======= end of job =======")
    print()


def dispatch(args, job_name, tag, fli_cmd, *, use_gpu=True, always_mpirun=False, env=None):
    """Dispatch *fli_cmd* according to args.mode.

    Parameters
    ----------
    use_gpu:
        When False, sbatch omits ``--gres=gpu`` and ``-C constraint``, and
        the MPI task count is ``tasks_per_node * nodes`` rather than
        ``gpus_per_node * nodes``.
    always_mpirun:
        When True, local mode always wraps with ``mpirun`` regardless of
        the GPU/task count (used by dorian-rt which is always MPI).
    env:
        Optional environment dict passed to ``subprocess.run``.
    """
    tpn, cpt, total_gpus = _slurm_params(args)

    if args.mode == "dryrun":
        _print_dryrun(job_name, args, fli_cmd)
        return

    if args.mode == "local":
        n = total_gpus if use_gpu else tpn * args.nodes
        if always_mpirun or n > 1:
            prefix = ["mpirun", "-n", str(n), "--oversubscribe"]
        else:
            prefix = []
        subprocess.run(prefix + fli_cmd, check=True, env=env)
        return

    # ---- sbatch mode ----
    if not args.slurm_script:
        print("Error: --slurm-script is required when --mode=sbatch", file=sys.stderr)
        sys.exit(1)

    constraint = args.constraint
    sbatch = ["sbatch", f"--account={args.account}"]
    if use_gpu and constraint and constraint != "cpu":
        sbatch += ["-C", constraint]
        sbatch += [f"--gres=gpu:{args.gpus_per_node}"]
    sbatch += [
        f"--time={args.time_limit}",
        f"--cpus-per-task={cpt}",
        f"--nodes={args.nodes}",
        f"--tasks-per-node={tpn}",
        f"--qos={args.qos}",
        f"--job-name={job_name}",
        f"--output={args.output_logs}/%x_%j.out",
        f"--error={args.output_logs}/%x_%j.err",
        os.path.expandvars(args.slurm_script),
        tag,
    ]
    subprocess.run(sbatch + fli_cmd, check=True, env=env)
