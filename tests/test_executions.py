import numpy as np
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from app.models import build_cnn
from app.state import DEFAULTS
from app.persistence import execution_record, save_execution, saved_executions, latest_training

HISTORY = {"loss": [1.0], "accuracy": [0.5], "val_loss": [1.2], "val_accuracy": [0.4]}


def test_distinct_runs_and_immutable_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("app.persistence.SAVED_MODELS", tmp_path)
    config = dict(DEFAULTS)
    model = build_cnn()
    first = execution_record(config, HISTORY, 2.5, 100)
    second = execution_record(config, HISTORY, 3.0, 100)
    config["epochs"] = 99
    assert first["training"]["epochs_requested"] == 3
    assert first["training"]["training_images"] == 85
    save_execution(first, model, "Baseline")
    save_execution(second, model, "Repetição")
    assert len(saved_executions("MNIST")) == 2
    assert saved_executions("CIFAR-10") == []
    with pytest.raises(ValueError, match="já foi salva"):
        save_execution(first, model, "Não sobrescrever")
    restored, errors = latest_training("MNIST")
    assert not errors
    assert restored[2] == HISTORY


def test_manual_save_only_and_retention_across_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr("app.persistence.SAVED_MODELS", tmp_path)
    images = np.zeros((20, 28, 28, 1), dtype=np.float32)
    monkeypatch.setattr("app.ui.training.cached_dataset", lambda name: (images, np.zeros(20), images, np.zeros(20)))
    def quick_training(config, loader, callback):
        loader(config["dataset"])
        return build_cnn(), {key: list(values) for key, values in HISTORY.items()}
    monkeypatch.setattr("app.ui.training.train_model", quick_training)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=60)
    app.secrets["app"] = {"allow_training": True}
    app.run()
    app.button(key="nav_Treinar").click().run()
    assert app.number_input(key="control_seed").value == 42
    app.number_input(key="control_seed").set_value(123).run()
    next(button for button in app.button if button.label == "Iniciar treinamento").click().run()
    assert not app.exception
    assert not list(tmp_path.rglob("*.keras"))
    run_id = next(iter(app.session_state["session_runs"]))
    app.number_input(key="control_epochs").set_value(8).run()
    app.number_input(key="control_seed").set_value(999).run()
    app.button(key="nav_Construir").click().run()
    app.selectbox(key="control_dataset").select("CIFAR-10").run()
    app.button(key="nav_Treinar").click().run()
    assert run_id in app.session_state["session_runs"]
    app.text_input(key="run_name_" + run_id).set_value("Meu MNIST").run()
    app.button(key="save_execution").click().run()
    assert not app.exception
    records = saved_executions("MNIST")
    assert len(records) == 1
    assert records[0][1]["training"]["epochs_requested"] == 3
    assert records[0][1]["training"]["total_images"] == 20
    assert records[0][1]["training"]["seed"] == 123
    assert app.session_state["session_runs"][run_id]["saved"]
    app.button(key="nav_Construir").click().run()
    app.selectbox(key="control_dataset").select("MNIST").run()
    app.button(key="nav_Treinar").click().run()
    assert app.session_state["trained"]
    assert app.selectbox(key="saved_run_selector").value
    next(button for button in app.button if button.label == "Carregar treino salvo").click().run()
    assert not app.exception
    assert app.session_state["history"] == HISTORY
