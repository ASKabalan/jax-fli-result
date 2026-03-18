"""Top-level argparse entry point for the launcher package."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="launcher",
        description="jax-fli job launcher",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    from launcher import simulate, grid, samples, infer, born_rt, extract, dorian_rt

    for mod in [simulate, grid, samples, infer, born_rt, extract, dorian_rt]:
        mod.add_subparser(sub)

    args = parser.parse_args()
    args.func(args)
