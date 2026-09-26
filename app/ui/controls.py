"""Widget adapters for centralized session transitions."""
import streamlit as st
from app import state


def navigate(stage):
    state.navigate(st.session_state, stage)


def remember(name):
    state.remember(st.session_state, name)


def keep_upload():
    state.keep_upload(st.session_state)


def clear_upload():
    state.clear_upload(st.session_state)


def control(kind, label, name, **kwargs):
    key = "control_" + name
    st.session_state[key] = st.session_state.settings[name]
    return getattr(st, kind)(label, key=key, on_change=remember, args=(name,), **kwargs)


