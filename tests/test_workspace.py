"""Behavior checks for the dock and contextual controls; no downloads required."""
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from app.models import build_cnn
from app.visualization import grad_cam, feature_maps, architecture_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setattr("app.persistence.SAVED_MODELS", tmp_path)


def test_dataset_restores_latest_training(tmp_path):
    import json
    import os
    from app.state import DEFAULTS, architecture_config, configuration_id
    from app.persistence import latest_training

    history = {key: [0.5] for key in ("loss", "val_loss", "accuracy", "val_accuracy")}
    for index, filters in enumerate((16, 64)):
        config = architecture_config(dict(DEFAULTS, dataset="CIFAR-10", filters=filters))
        model = build_cnn(filters=filters, input_shape=(32, 32, 3))
        path = tmp_path / (configuration_id(config) + ".keras")
        model.save(path)
        path.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}))
        os.utime(path, (100 + index, 100 + index))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60).run()
    assert not app.session_state["trained"]
    app.selectbox(key="control_dataset").select("CIFAR-10").run()
    assert not app.exception
    assert app.session_state["trained"]
    assert app.select_slider(key="control_filters").value == 64
    assert app.session_state["history"] == history
    image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    np.testing.assert_allclose(model(image), app.session_state["model"](image))
    restored = app.session_state["model"]
    app.selectbox(key="control_view").select("Grafo").run()
    assert app.session_state["model"] is restored
    app.select_slider(key="control_filters").set_value(32).run()
    assert not app.session_state["trained"]
    app.selectbox(key="control_dataset").select("MNIST").run()
    assert not app.session_state["trained"]
    app.selectbox(key="control_dataset").select("CIFAR-10").run()
    assert app.session_state["trained"]
    assert app.select_slider(key="control_filters").value == 64
    path.write_bytes(b"broken checkpoint")
    recovered, errors = latest_training("CIFAR-10")
    assert errors and recovered[1]["filters"] == 16


def test_startup_restores_training(tmp_path):
    import json
    from app.state import DEFAULTS, architecture_config, configuration_id
    config = architecture_config(DEFAULTS)
    path = tmp_path / (configuration_id(config) + ".keras")
    build_cnn().save(path)
    history = {key: [0.5] for key in ("loss", "val_loss", "accuracy", "val_accuracy")}
    path.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}))
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60)
    app.secrets["app"] = {"allow_training": False}
    app.run()
    assert not app.exception
    assert app.session_state["trained"]
    assert app.session_state["history"] == history
    app.button(key="nav_Treinar").click().run()
    assert not app.exception
    assert all(button.label != "Iniciar treinamento" for button in app.button)
    assert app.session_state["history"] == history


def test_cloud_disables_training_without_model(monkeypatch):
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Training must not run")
    monkeypatch.setattr("tensorflow.keras.Model.fit", unexpected_fit)
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60)
    app.secrets["app"] = {"allow_training": False}
    app.run()
    app.button(key="nav_Treinar").click().run()
    assert not app.exception
    assert all(button.label != "Iniciar treinamento" for button in app.button)
    assert not app.number_input
    assert any("Não há modelo treinado" in msg.value for msg in app.info)
    app.button(key="nav_Explicar").click().run()
    assert not app.exception
    assert any("treinamento está desabilitado" in msg.value for msg in app.info)


def test_dock_preserves_settings_and_model_without_training(monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("Opening the builder must not load a dataset")
    monkeypatch.setattr("app.datasets.load_dataset", unexpected_download)
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.session_state["trained"]
    app.select_slider(key="control_filters").set_value(64).run()
    assert not app.exception
    model = app.session_state["model"]
    app.button(key="nav_Treinar").click().run()
    assert not app.exception
    assert app.session_state["model"] is model
    assert len(app.slider) == 0  # Architecture sliders are not present during training.
    app.number_input(key="control_epochs").set_value(7).run()
    app.button(key="nav_Explicar").click().run()
    assert not app.exception
    assert any("Treine o modelo" in message.value for message in app.info)
    app.button(key="nav_Construir").click().run()
    assert app.select_slider(key="control_filters").value == 64
    app.button(key="nav_Treinar").click().run()
    assert app.number_input(key="control_epochs").value == 7
    assert app.session_state["model"] is model


def test_architecture_change_invalidates_training():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60).run()
    old_model = app.session_state["model"]
    app.session_state["trained"] = True
    app.session_state["ever_trained"] = True
    app.selectbox(key="control_dataset").select("CIFAR-10").run()
    assert not app.exception
    assert not app.session_state["trained"]
    assert app.session_state["model"] is not old_model
    assert app.session_state["model"].input_shape == (None, 32, 32, 3)
    app.slider(key="control_num_conv").set_value(5).run()
    assert not app.exception
    assert any("dimensão inválida" in error.value for error in app.error)


def test_feature_maps_and_gradcam_are_finite_and_preserve_predictions():
    model = build_cnn(num_conv=1, filters=16)
    image = np.random.default_rng(42).random((1, 28, 28, 1), dtype=np.float32)
    before = np.asarray(model(image, training=False))
    maps = feature_maps(model, image, "conv_1")
    assert maps.shape == (26, 26, 16)
    heatmap, predicted = grad_cam(model, image, "conv_1")
    assert predicted == int(before[0].argmax())
    assert heatmap.shape == (26, 26)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    np.testing.assert_allclose(before, model(image, training=False))
    for variable in model.weights:
        variable.assign(np.zeros(variable.shape))
    empty, _ = grad_cam(model, image, "conv_1")
    assert np.isfinite(empty).all() and not empty.any()


@pytest.mark.parametrize("view,sizing", [
    ("Camadas", "accurate"), ("Camadas", "balanced"), ("Camadas", "capped"),
    ("Camadas", "logarithmic"), ("Camadas", "relative"), ("Grafo", "balanced"),
    ("Functional View", "balanced"), ("LeNet View", "balanced"),
])
def test_architecture_renderers(view, sizing):
    model = build_cnn(num_conv=1, filters=16)
    image = np.random.default_rng(42).random((1, 28, 28, 1), dtype=np.float32)
    before = np.asarray(model(image, training=False))
    shapes_before = [getattr(layer, "output_shape", None) for layer in model.layers]
    drawing = architecture_image(model, view=view, sizing=sizing)
    assert drawing.width > 0 and drawing.height > 0
    assert [getattr(layer, "output_shape", None) for layer in model.layers] == shapes_before
    np.testing.assert_allclose(before, model(image, training=False))


def test_new_views_preserve_model_and_training():
    app = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=60).run()
    model = app.session_state["model"]
    app.session_state["trained"] = True
    for view in ("Functional View", "LeNet View", "Camadas"):
        app.selectbox(key="control_view").select(view).run()
        assert not app.exception
        assert not app.warning
        assert app.session_state["model"] is model
        assert app.session_state["trained"]
        assert app.selectbox(key="control_sizing").disabled == (view != "Camadas")


def test_invalid_spatial_configuration():
    with pytest.raises(ValueError, match="bloco"):
        build_cnn(num_conv=5)


def test_real_training_step_without_network():
    model = build_cnn(num_conv=1, filters=16, pooling="AveragePooling2D", dense_units=32)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    image = np.random.default_rng(7).random((4, 28, 28, 1), dtype=np.float32)
    result = model.train_on_batch(image, np.arange(4), return_dict=True)
    assert np.isfinite(result["loss"])
