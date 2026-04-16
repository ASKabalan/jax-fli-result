"""Simulation settings form — mesh, box, observer, output target, etc.

Used by Simulate (grid=True) and Samples/Infer (grid=False).
"""
from __future__ import annotations

import streamlit as st

from app.components.dynamic_list import render_dynamic_triple_list


def render_simulation_settings(
    prefix: str = "",
    defaults: dict | None = None,
    grid: bool = True,
    px: int = 1,
    py: int = 1,
) -> dict:
    """Render the simulation settings form.

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.
    grid:
        When True (Simulate page), mesh_size/box_size use
        render_dynamic_triple_list and seed uses render_dynamic_list.
        When False (Samples/Infer), single MX/MY/MZ and BX/BY/BZ triplets
        and seed is a single integer.
    px, py:
        Process grid dimensions — used for halo-size validation.

    Returns
    -------
    dict with keys:
        mesh_size, box_size, observer_position, halo_multiplier, seed,
        enable_x64, scheme, paint_nside, kernel_width_arcmin.
        Grid mode only: nside, flatsky_npix, field_size, density (bool).
    """
    from app.components.dynamic_list import render_dynamic_list

    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Simulation Settings")

        mh1, mh2 = st.columns([2, 1])
        with mh1:
            st.write("**Mesh sizes**" if grid else "**mesh_size**")
        with mh2:
            halo_multiplier = st.number_input(
                "Halo multiplier" if grid else "Halo mult.",
                min_value=0.0,
                value=float(defaults.get("halo_multiplier", 0.5)),
                step=0.05,
                format="%.2f",
                key=f"{prefix}halo_multiplier",
                help="Halo size = local_mesh × halo_multiplier",
            )

        if grid:
            mesh_size = render_dynamic_triple_list(
                "mesh_size",
                f"{prefix}mesh_size",
                defaults.get("mesh_size_triples", [(64, 64, 64), (32, 32, 32)]),
                cast_fn=int,
            )
            st.write("**Box sizes**")
            box_size = render_dynamic_triple_list(
                "box_size",
                f"{prefix}box_size",
                defaults.get("box_size_triples", [(1000, 1000, 1000)]),
                cast_fn=float,
            )
        else:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                mx = st.number_input(
                    "MX", value=int(defaults.get("mx", 64)), key=f"{prefix}mx"
                )
            with mc2:
                my = st.number_input(
                    "MY", value=int(defaults.get("my", 64)), key=f"{prefix}my"
                )
            with mc3:
                mz = st.number_input(
                    "MZ", value=int(defaults.get("mz", 64)), key=f"{prefix}mz"
                )
            mesh_size = [mx, my, mz]

            st.write("**box_size**")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                bx = st.number_input(
                    "BX", value=float(defaults.get("bx", 250.0)), key=f"{prefix}bx"
                )
            with bc2:
                by = st.number_input(
                    "BY", value=float(defaults.get("by", 250.0)), key=f"{prefix}by"
                )
            with bc3:
                bz = st.number_input(
                    "BZ", value=float(defaults.get("bz", 250.0)), key=f"{prefix}bz"
                )
            box_size = [bx, by, bz]

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input(
                "OX",
                value=float(defaults.get("obs_x", 0.5)),
                format="%.2f",
                key=f"{prefix}obs_x",
            )
        with oc2:
            obs_y = st.number_input(
                "OY",
                value=float(defaults.get("obs_y", 0.5)),
                format="%.2f",
                key=f"{prefix}obs_y",
            )
        with oc3:
            obs_z = st.number_input(
                "OZ",
                value=float(defaults.get("obs_z", 0.5)),
                format="%.2f",
                key=f"{prefix}obs_z",
            )

        if grid:
            seed = render_dynamic_list("Seed", f"{prefix}seed", ["0"], cast_fn=str)
        else:
            seed = st.number_input(
                "seed",
                min_value=0,
                value=int(defaults.get("seed", 0)),
                key=f"{prefix}seed",
            )

        st.divider()
        scheme = st.selectbox(
            "scheme",
            ["ngp", "bilinear", "rbf_neighbor"],
            index=1,
            key=f"{prefix}scheme",
        )
        use_paint = st.checkbox(
            "Set paint_nside", value=False, key=f"{prefix}use_paint_nside"
        )
        paint_nside = None
        if use_paint:
            paint_nside = st.number_input(
                "paint_nside", min_value=1, value=64, key=f"{prefix}paint_nside"
            )
        kernel_width_arcmin = None
        if scheme == "rbf_neighbor":
            with st.expander("RBF parameters"):
                activate_rbf = st.checkbox(
                    "Activate kernel width",
                    value=False,
                    key=f"{prefix}activate_rbf",
                )
                if activate_rbf:
                    kernel_width_arcmin = st.number_input(
                        "kernel_width_arcmin",
                        value=5.0,
                        min_value=0.01,
                        format="%.2f",
                        key=f"{prefix}kernel_width_arcmin",
                    )

        enable_x64 = st.checkbox(
            "enable_x64",
            value=bool(defaults.get("enable_x64", False)),
            key=f"{prefix}enable_x64",
        )

        # Halo validation (grid=True uses flat mesh list; grid=False uses single triplet)
        if grid:
            _mesh_triples = [mesh_size[i : i + 3] for i in range(0, len(mesh_size), 3)]
            for _triple in _mesh_triples:
                if len(_triple) < 3:
                    continue
                _mx, _my, _ = _triple
                _exp_x = (_mx / px) * (1 + halo_multiplier)
                _exp_y = (_my / py) * (1 + halo_multiplier)
                if (
                    abs(_exp_x - round(_exp_x)) > 1e-9
                    or abs(_exp_y - round(_exp_y)) > 1e-9
                ):
                    st.error(
                        f"Mesh {_mx}×{_my}: local_mesh × (1 + halo_multiplier) = "
                        f"{_exp_x:.3f}, {_exp_y:.3f} — must be integer"
                    )
        else:
            _exp_x = (mesh_size[0] / px) * (1 + halo_multiplier)
            _exp_y = (mesh_size[1] / py) * (1 + halo_multiplier)
            if abs(_exp_x - round(_exp_x)) > 1e-9 or abs(_exp_y - round(_exp_y)) > 1e-9:
                st.error(
                    f"Mesh {mesh_size[0]}×{mesh_size[1]}: local_mesh × (1+halo) = "
                    f"{_exp_x:.3f}, {_exp_y:.3f} — must be integer"
                )

    return {
        "mesh_size": mesh_size,
        "box_size": box_size,
        "observer_position": [obs_x, obs_y, obs_z],
        "halo_multiplier": halo_multiplier,
        "seed": seed,
        "enable_x64": enable_x64,
        "scheme": scheme,
        "paint_nside": paint_nside,
        "kernel_width_arcmin": kernel_width_arcmin,
    }
