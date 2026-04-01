"""SLURM / cluster argument form — mirrors parser.add_slurm_args."""
from __future__ import annotations

import streamlit as st


def render_slurm_form(defaults: dict | None = None, prefix: str = "") -> dict:
    defaults = defaults or {}
    with st.container(border=True):
        st.subheader("SLURM / Cluster")

        mode = st.selectbox(
            "Mode",
            ["dryrun", "local", "sbatch"],
            index=["dryrun", "local", "sbatch"].index(defaults.get("mode", "dryrun")),
            key=f"{prefix}mode",
        )
        account = st.text_input("Account", value=defaults.get("account", "XXX"), key=f"{prefix}account")
        constraint = st.text_input("Constraint", value=defaults.get("constraint", "h100"), key=f"{prefix}constraint")
        qos = st.text_input("QoS", value=defaults.get("qos", "qos_gpu_h100-t3"), key=f"{prefix}qos")
        time_limit = st.text_input("Time limit", value=defaults.get("time_limit", "00:30:00"), key=f"{prefix}time_limit")
        output_logs = st.text_input("Output logs dir", value=defaults.get("output_logs", "SLURM_LOGS"), key=f"{prefix}output_logs")

        c1, c2 = st.columns(2)
        with c1:
            gpus_per_node = st.number_input("GPUs per node", min_value=0, value=defaults.get("gpus_per_node", 4), key=f"{prefix}gpus_per_node")
        with c2:
            cpus_per_node = st.number_input("CPUs per node", min_value=1, value=defaults.get("cpus_per_node", 16), key=f"{prefix}cpus_per_node")

        c3, c4 = st.columns(2)
        with c3:
            tasks_per_node = st.number_input(
                "Tasks per node (0 = auto from GPUs)",
                min_value=0,
                value=defaults.get("tasks_per_node", 0),
                key=f"{prefix}tasks_per_node",
            )
        with c4:
            nodes = st.number_input("Nodes", min_value=1, value=defaults.get("nodes", 4), key=f"{prefix}nodes")

        slurm_script = None
        if mode == "sbatch":
            slurm_script = st.text_input("SLURM script path", value=defaults.get("slurm_script", ""), key=f"{prefix}slurm_script")

        st.write("**pdim** (process grid)")
        pc1, pc2 = st.columns(2)
        pdim_default = defaults.get("pdim", [16, 1])
        with pc1:
            px = st.number_input("PX", min_value=1, value=pdim_default[0], key=f"{prefix}pdim_x")
        with pc2:
            py = st.number_input("PY", min_value=1, value=pdim_default[1], key=f"{prefix}pdim_y")

        return {
            "mode": mode,
            "account": account,
            "constraint": constraint,
            "qos": qos,
            "time_limit": time_limit,
            "output_logs": output_logs,
            "gpus_per_node": gpus_per_node,
            "cpus_per_node": cpus_per_node,
            "tasks_per_node": tasks_per_node if tasks_per_node > 0 else None,
            "nodes": nodes,
            "slurm_script": slurm_script,
            "pdim": [px, py],
        }
