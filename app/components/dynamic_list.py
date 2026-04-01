"""Reusable dynamic-list widgets for Streamlit forms."""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Callbacks (executed before rerun so state is consistent)
# ---------------------------------------------------------------------------

def _add_entry(key: str, data_key: str, counter_key: str, default_val: str) -> None:
    uid = st.session_state[counter_key]
    st.session_state[counter_key] = uid + 1
    st.session_state[data_key].append(uid)
    st.session_state[f"_dl_{key}_v_{uid}"] = str(default_val)


def _remove_entry(data_key: str, uid: int) -> None:
    st.session_state[data_key].remove(uid)


def _add_triple(key: str, data_key: str, counter_key: str, default_triple: tuple) -> None:
    uid = st.session_state[counter_key]
    st.session_state[counter_key] = uid + 1
    st.session_state[data_key].append(uid)
    st.session_state[f"_dtl_{key}_x_{uid}"] = str(default_triple[0])
    st.session_state[f"_dtl_{key}_y_{uid}"] = str(default_triple[1])
    st.session_state[f"_dtl_{key}_z_{uid}"] = str(default_triple[2])


def _remove_triple(data_key: str, uid: int) -> None:
    st.session_state[data_key].remove(uid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_dynamic_list(label: str, key: str, defaults: list, cast_fn=str) -> list:
    """Render a dynamic list of single-value entries.

    Returns list[T] where T is determined by *cast_fn*.
    """
    data_key = f"_dl_{key}"
    counter_key = f"_dl_{key}_ctr"

    if data_key not in st.session_state:
        st.session_state[data_key] = list(range(len(defaults)))
        st.session_state[counter_key] = len(defaults)
        for i, d in enumerate(defaults):
            st.session_state[f"_dl_{key}_v_{i}"] = str(d)

    st.write(f"**{label}**")

    for uid in st.session_state[data_key]:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.text_input(label, key=f"_dl_{key}_v_{uid}", label_visibility="collapsed")
        with c2:
            st.button(
                "\u2715", key=f"_dl_{key}_rm_{uid}",
                on_click=_remove_entry, args=(data_key, uid),
            )

    default_val = defaults[0] if defaults else ""
    st.button(
        f"\u002B Add {label}", key=f"_dl_{key}_add",
        on_click=_add_entry, args=(key, data_key, counter_key, default_val),
    )

    result = []
    for uid in st.session_state[data_key]:
        raw = st.session_state.get(f"_dl_{key}_v_{uid}", "")
        try:
            result.append(cast_fn(raw))
        except (ValueError, TypeError):
            pass
    return result


def render_dynamic_triple_list(label: str, key: str, defaults: list[tuple], cast_fn=int) -> list:
    """Render a dynamic list of (X, Y, Z) triple entries.

    *defaults* is a list of 3-tuples, e.g. ``[(64,64,64), (32,32,32)]``.
    Returns a flat list ``[x1,y1,z1,x2,y2,z2,...]``.
    """
    data_key = f"_dtl_{key}"
    counter_key = f"_dtl_{key}_ctr"

    if data_key not in st.session_state:
        st.session_state[data_key] = list(range(len(defaults)))
        st.session_state[counter_key] = len(defaults)
        for i, triple in enumerate(defaults):
            st.session_state[f"_dtl_{key}_x_{i}"] = str(triple[0])
            st.session_state[f"_dtl_{key}_y_{i}"] = str(triple[1])
            st.session_state[f"_dtl_{key}_z_{i}"] = str(triple[2])

    st.write(f"**{label}**")

    for uid in st.session_state[data_key]:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            st.text_input("X", key=f"_dtl_{key}_x_{uid}", label_visibility="collapsed")
        with c2:
            st.text_input("Y", key=f"_dtl_{key}_y_{uid}", label_visibility="collapsed")
        with c3:
            st.text_input("Z", key=f"_dtl_{key}_z_{uid}", label_visibility="collapsed")
        with c4:
            st.button(
                "\u2715", key=f"_dtl_{key}_rm_{uid}",
                on_click=_remove_triple, args=(data_key, uid),
            )

    default_triple = defaults[0] if defaults else (0, 0, 0)
    st.button(
        f"\u002B Add {label}", key=f"_dtl_{key}_add",
        on_click=_add_triple, args=(key, data_key, counter_key, default_triple),
    )

    result = []
    for uid in st.session_state[data_key]:
        for axis in ("x", "y", "z"):
            raw = st.session_state.get(f"_dtl_{key}_{axis}_{uid}", "0")
            try:
                result.append(cast_fn(raw))
            except (ValueError, TypeError):
                result.append(cast_fn(0))
    return result
