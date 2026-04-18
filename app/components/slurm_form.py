"""SLURM / cluster argument form — mirrors parser.add_slurm_args."""
from __future__ import annotations

import streamlit as st


def render_slurm_form(
    defaults: dict | None = None,
    prefix: str = "",
    show_tasks_per_node: bool = False,
    show_pdim: bool = True,
) -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("SLURM / Cluster")

        # ── Dispatch mode ────────────────────────────────────────────────────
        mode = st.selectbox(
            "Dispatch mode",
            ["dryrun", "local", "sbatch"],
            index=["dryrun", "local", "sbatch"].index(defaults.get("mode", "dryrun")),
            key=f"{prefix}mode",
        )

        # ── Cluster settings ─────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            account = st.text_input(
                "Account", value=defaults.get("account", "XXX"), key=f"{prefix}account"
            )
        with c2:
            constraint = st.text_input(
                "Constraint",
                value=defaults.get("constraint", "h100"),
                key=f"{prefix}constraint",
            )

        c3, c4 = st.columns(2)
        with c3:
            qos = st.text_input(
                "QoS", value=defaults.get("qos", "qos_gpu_h100-t3"), key=f"{prefix}qos"
            )
        with c4:
            time_limit = st.text_input(
                "Time limit",
                value=defaults.get("time_limit", "00:30:00"),
                key=f"{prefix}time_limit",
            )

        # ── Nodes / GPUs ──────────────────────────────────────────────────────
        c5, c6 = st.columns(2)
        with c5:
            nodes = st.number_input(
                "Nodes",
                min_value=1,
                value=defaults.get("nodes", 4),
                key=f"{prefix}nodes",
            )
        with c6:
            gpus_per_node = st.number_input(
                "GPUs per node",
                min_value=0,
                value=defaults.get("gpus_per_node", 4),
                key=f"{prefix}gpus_per_node",
                help="tasks-per-node will equal this value (one task per GPU is the only supported mode)",
            )

        cpus_per_node = st.number_input(
            "CPUs per node",
            min_value=1,
            value=defaults.get("cpus_per_node", 16),
            key=f"{prefix}cpus_per_node",
        )

        # tasks_per_node: shown only when explicitly requested (e.g. Dorian CPU jobs)
        tasks_per_node = None
        if show_tasks_per_node:
            tasks_per_node_val = defaults.get("tasks_per_node", 4) or 4
            tasks_per_node = st.number_input(
                "Tasks per node",
                min_value=1,
                value=tasks_per_node_val,
                key=f"{prefix}tasks_per_node",
                help="Number of MPI tasks per node (CPU jobs only — GPU jobs derive this from GPUs per node)",
            )

        # ── pdim with validation (GPU jobs only) ─────────────────────────────
        px = py = 1  # defaults used when pdim section is hidden
        if show_pdim:
            st.write("**pdim** (process grid)")
            pdim_default = defaults.get("pdim", [16, 1])
            pc1, pc2 = st.columns(2)
            with pc1:
                px = st.number_input(
                    "PX", min_value=1, value=pdim_default[0], key=f"{prefix}pdim_x"
                )
            with pc2:
                py = st.number_input(
                    "PY", min_value=1, value=pdim_default[1], key=f"{prefix}pdim_y"
                )

            pdim_product = px * py
            node_gpu_product = nodes * gpus_per_node
            if pdim_product != node_gpu_product:
                st.error(
                    f"pdim product ({px} × {py} = {pdim_product}) ≠ "
                    f"nodes × GPUs/node ({nodes} × {gpus_per_node} = {node_gpu_product})"
                )

        # ── sbatch-only fields ────────────────────────────────────────────────
        output_logs = "SLURM_LOGS"
        slurm_script = None
        if mode == "sbatch":
            output_logs = st.text_input(
                "Output logs dir",
                value=defaults.get("output_logs", "SLURM_LOGS"),
                key=f"{prefix}output_logs",
            )
            slurm_script_raw = st.text_input(
                "SLURM script path (leave empty to omit)",
                value=defaults.get("slurm_script", ""),
                key=f"{prefix}slurm_script",
            )
            slurm_script = (
                slurm_script_raw
                if slurm_script_raw and slurm_script_raw.strip()
                else None
            )

        result = {
            "mode": mode,
            "account": account,
            "constraint": constraint if constraint and constraint.strip() else None,
            "qos": qos,
            "time_limit": time_limit,
            "output_logs": output_logs,
            "gpus_per_node": gpus_per_node,
            "cpus_per_node": cpus_per_node,
            "tasks_per_node": tasks_per_node,
            "nodes": nodes,
            "slurm_script": slurm_script,
        }
        if show_pdim:
            result["pdim"] = [px, py]
        return result
