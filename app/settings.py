"""Deployment settings, independent of UI session state."""
import streamlit as st


def training_allowed():
    """Local installations without a secrets file allow training by default."""
    try:
        value = st.secrets.get("app", {}).get("allow_training", True)
    except FileNotFoundError:
        return True
    # Only TOML booleans enable training; a mistyped value fails closed.
    return value is True
