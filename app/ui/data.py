import streamlit as st
from app.datasets import load_dataset

@st.cache_data(max_entries=1, show_spinner="Carregando dataset…")
def cached_dataset(name):
    return load_dataset(name)
