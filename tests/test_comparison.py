from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from app.persistence import execution_record
from app.state import DEFAULTS
from app.ui.comparison import summary


def record(epochs, total=100):
    history = {"loss": [0.8] * epochs, "accuracy": [0.7] * epochs,
               "val_loss": [0.9] * epochs, "val_accuracy": [0.8] + [0.6] * (epochs - 1)}
    return dict(execution_record(DEFAULTS, history, 4.0, total), name="Mesmo nome", source="Salva")


def test_summary_uses_final_and_best_separately():
    frame = summary([record(3), record(1)])
    assert frame.iloc[0]["Acurácia final de validação (%)"] == pytest.approx(60)
    assert frame.iloc[0]["Melhor acurácia de validação (%)"] == pytest.approx(80)
    assert frame["Execução"].nunique() == 2


def test_comparison_preserves_model_and_filters_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr("app.persistence.SAVED_MODELS", tmp_path)
    first, second = record(3), record(1, total=200)
    monkeypatch.setattr("app.ui.comparison.saved_executions", lambda dataset:
                        [(None, first), (None, second)] if dataset == "MNIST" else [])
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=60).run()
    model = app.session_state["model"]
    app.button(key="nav_Comparar").click().run()
    assert not app.exception
    assert app.session_state["model"] is model
    assert len(app.dataframe) == 1
    assert any("divisões" in message.value for message in app.warning)
    app.selectbox(key="comparison_metric").select("Loss").run()
    assert not app.exception
    app.multiselect(key="compare_runs_MNIST").set_value([first["run_id"]]).run()
    assert any("pelo menos duas" in message.value for message in app.info)
    app.selectbox(key="comparison_dataset").select("CIFAR-10").run()
    assert any("Nenhuma execução" in message.value for message in app.info)
    assert app.session_state["model"] is model
    assert app.session_state["settings"]["dataset"] == "MNIST"
