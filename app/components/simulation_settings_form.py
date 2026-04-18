"""Simulation settings form — mesh, box, observer, output target, etc."""
from __future__ import annotations

import streamlit as st


def render_simulation_settings(
    prefix: str = "",
    defaults: dict | None = None,
    px: int = 1,
    py: int = 1,
) -> dict:
    """Render the simulation settings form (single-point, no gridding).

    Parameters
    ----------
    prefix:
        Streamlit key prefix for namespacing.
    defaults:
        Optional overrides for default values.
    px, py:
        Process grid dimensions — used for halo-size validation.
    show_output_target:
        When True, render an output-type radio (Spherical / Flat sky /
        Density / Particles) with conditional nside, flatsky_npix, field_size
        inputs.  scheme and paint_nside are rendered inside the Spherical
        branch only.
    show_nside:
        When True (and show_output_target is False), render a standalone
        nside number input.  Used by Samples (always spherical).
    """
    defaults = defaults or {}

    with st.container(border=True):
        st.subheader("Simulation Settings")

        mh1, mh2 = st.columns([2, 1])
        with mh1:
            st.write("**mesh_size**")
        with mh2:
            halo_multiplier = st.number_input(
                "Halo multiplier",
                min_value=0.0,
                value=float(defaults.get("halo_multiplier", 0.5)),
                step=0.05,
                format="%.2f",
                key=f"{prefix}halo_multiplier",
                help="Halo size = local_mesh × halo_multiplier",
            )

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            mx = st.number_input("MX", value=int(defaults.get("mx", 64)), key=f"{prefix}mx")
        with mc2:
            my = st.number_input("MY", value=int(defaults.get("my", 64)), key=f"{prefix}my")
        with mc3:
            mz = st.number_input("MZ", value=int(defaults.get("mz", 64)), key=f"{prefix}mz")
        mesh_size = [mx, my, mz]

        st.write("**box_size**")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bx = st.number_input("BX", value=float(defaults.get("bx", 250.0)), key=f"{prefix}bx")
        with bc2:
            by = st.number_input("BY", value=float(defaults.get("by", 250.0)), key=f"{prefix}by")
        with bc3:
            bz = st.number_input("BZ", value=float(defaults.get("bz", 250.0)), key=f"{prefix}bz")
        box_size = [bx, by, bz]

        st.write("**Observer position**")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            obs_x = st.number_input("OX", value=float(defaults.get("obs_x", 0.5)), format="%.2f", key=f"{prefix}obs_x")
        with oc2:
            obs_y = st.number_input("OY", value=float(defaults.get("obs_y", 0.5)), format="%.2f", key=f"{prefix}obs_y")
        with oc3:
            obs_z = st.number_input("OZ", value=float(defaults.get("obs_z", 0.5)), format="%.2f", key=f"{prefix}obs_z")

        seed = st.number_input(
            "seed",
            min_value=0,
            value=int(defaults.get("seed", 0)),
            key=f"{prefix}seed",
        )

        st.divider()

        # ── Projection / output-target section ────────────────────────────────
        st.write("**Projection / output target**")
        output_target = None
        nside = None
        flatsky_npix = None
        field_size = None
        density = False
        scheme = defaults.get("scheme", "bilinear")
        paint_nside = None
        kernel_width_arcmin = None

        _out_options = ["Spherical (nside)", "Flat sky", "Density", "Particles"]
        _default_out = defaults.get("output_target", "Spherical (nside)")
        _out_idx = _out_options.index(_default_out) if _default_out in _out_options else 0
        output_target = st.radio(
            "Output target",
            _out_options,
            index=_out_idx,
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}output_target",
        )

        if output_target == "Spherical (nside)":
            nside = st.number_input(
                "nside",
                min_value=1,
                value=int(defaults.get("nside", 64)),
                key=f"{prefix}nside",
            )
            _scheme_opts = ["ngp", "bilinear", "rbf_neighbor"]
            scheme = st.selectbox(
                "scheme",
                _scheme_opts,
                index=_scheme_opts.index(defaults.get("scheme", "bilinear")),
                key=f"{prefix}scheme",
            )
            paint_nside = st.number_input(
                "paint_nside",
                min_value=1,
                value=int(defaults.get("paint_nside", 64)),
                key=f"{prefix}paint_nside",
            )
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
                            value=float(defaults.get("kernel_width_arcmin", 5.0)),
                            min_value=0.01,
                            format="%.2f",
                            key=f"{prefix}kernel_width_arcmin",
                        )

        elif output_target == "Flat sky":
            st.write("**Pixels (H × W)**")
            fp1, fp2 = st.columns(2)
            with fp1:
                _fp_h = st.number_input("H", min_value=1, value=64, key=f"{prefix}flatsky_h")
            with fp2:
                _fp_w = st.number_input("W", min_value=1, value=64, key=f"{prefix}flatsky_w")
            flatsky_npix = [_fp_h, _fp_w]

            st.write("**Field size (H × W) deg**")
            ff1, ff2 = st.columns(2)
            with ff1:
                _ff_h = st.number_input("H", min_value=1, value=10, key=f"{prefix}field_h")
            with ff2:
                _ff_w = st.number_input("W", min_value=1, value=10, key=f"{prefix}field_w")
            field_size = [_ff_h, _ff_w]
            if any(v > 10 for v in field_size):
                st.warning("Flat sky approximation is only reliable up to ~10 degrees.")

        elif output_target == "Density":
            density = True
        # Particles: no extra inputs needed

        enable_x64 = st.checkbox(
            "enable_x64",
            value=bool(defaults.get("enable_x64", False)),
            key=f"{prefix}enable_x64",
        )

        _exp_x = (mesh_size[0] / px) * (1 + halo_multiplier)
        _exp_y = (mesh_size[1] / py) * (1 + halo_multiplier)
        if abs(_exp_x - round(_exp_x)) > 1e-9 or abs(_exp_y - round(_exp_y)) > 1e-9:
            st.error(
                f"Mesh {mesh_size[0]}×{mesh_size[1]}: local_mesh × (1+halo) = "
                f"{_exp_x:.3f}, {_exp_y:.3f} — must be integer"
            )

    result = {
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
    if show_output_target:
        result.update(
            {
                "output_target": output_target,
                "nside": nside,
                "flatsky_npix": flatsky_npix,
                "field_size": field_size,
                "density": density,
            }
        )
    if show_nside and not show_output_target:
        result["nside"] = nside
    return result
