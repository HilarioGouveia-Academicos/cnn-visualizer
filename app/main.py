"""Application startup and stage routing."""
from pathlib import Path
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
st.set_page_config(page_title="CNN Learn", page_icon=":material/hub:",
                   layout="wide", initial_sidebar_state="expanded")

from app.settings import training_allowed
from app.state import initialize, restore_dataset, synchronize_model
from app.ui.navigation import STEPS, render_navigation
from app.ui import builder, training, exploration, explanation, comparison

ss = st.session_state
initialize(ss)
restore_dataset(ss)
allow_training = training_allowed()
train_clicked, load_clicked, cam_clicked = render_navigation(ss, allow_training)
stage = ss.stage
st.caption("CNN LEARN / LABORATÓRIO INTERATIVO")
heading, status_column = st.columns([3, 1], vertical_alignment="center")
with heading:
    st.title(f"{list(STEPS).index(stage) + 1}. {stage}")
    st.caption(STEPS[stage][1])
with status_column:
    status_slot = st.empty()

if stage == "Comparar":
    comparison.render(ss)
    st.stop()

try:
    synchronize_model(ss)
except ValueError as exc:
    status_slot.badge("Arquitetura inválida", color="red")
    st.error(str(exc))
    st.stop()

for message in ss.pop("restore_errors", []):
    st.warning(message)
if ss.get("restored_dataset"):
    st.caption(f'Treino salvo de {ss.restored_dataset} recuperado automaticamente, com arquitetura e métricas.')
if ss.trained:
    status_slot.badge("Treinado", icon=":material/check_circle:", color="green")
elif ss.ever_trained:
    status_slot.badge("Configuração alterada", color="orange")
else:
    status_slot.badge("Não treinado", color="gray")

if stage == "Construir":
    builder.render(ss)
elif stage == "Treinar":
    training.render(ss, allow_training, train_clicked, load_clicked, status_slot)
elif stage == "Explorar":
    exploration.render(ss)
elif stage == "Explicar":
    explanation.render(ss, allow_training, cam_clicked)
