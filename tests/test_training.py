import numpy as np
import pytest
from app.state import DEFAULTS
from app.training import train_model
from app.persistence import save_training, load_training


def test_seed_controls_initial_weights(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("app.training.training_allowed", lambda: True)
    # Capture the actual model initialization without spending epochs training.
    monkeypatch.setattr("tensorflow.keras.Model.fit", lambda *args, **kwargs:
                        SimpleNamespace(history={"loss": [1.0]}))
    images = np.zeros((12, 28, 28, 1), dtype=np.float32)
    loader = lambda name: (images, np.zeros(12), images, np.zeros(12))
    config = dict(DEFAULTS, num_conv=1, filters=16, seed=123)
    first, _ = train_model(config, loader)
    second, _ = train_model(config, loader)
    third, _ = train_model(dict(config, seed=124), loader)
    for left, right in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first.get_weights()[0], third.get_weights()[0])


@pytest.mark.parametrize("seed", [-1, 4294967296, 1.5, True])
def test_invalid_seed_is_rejected(monkeypatch, seed):
    monkeypatch.setattr("app.training.training_allowed", lambda: True)
    with pytest.raises(ValueError, match="semente"):
        train_model(dict(DEFAULTS, seed=seed), None)


def test_disabled_training_does_not_load_data(monkeypatch):
    monkeypatch.setattr("app.training.training_allowed", lambda: False)
    def unexpected_loader(name):
        pytest.fail("Blocked training must not load data")
    with pytest.raises(PermissionError):
        train_model(DEFAULTS, unexpected_loader)


def test_training_progress_and_checkpoint_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr("app.training.training_allowed", lambda: True)
    monkeypatch.setattr("app.persistence.SAVED_MODELS", tmp_path)
    images = np.random.default_rng(4).random((12, 28, 28, 1), dtype=np.float32)
    labels = np.arange(12) % 10
    config = dict(DEFAULTS, num_conv=1, filters=16, epochs=1, batch_size=4)
    progress = []
    model, history = train_model(config, lambda name: (images, labels, images, labels),
                                 lambda epoch, metrics: progress.append((epoch, dict(metrics))))
    assert progress[0][0] == 0
    assert progress[0][1] == history
    assert np.isfinite(history["loss"]).all()
    save_training(config, model, history)
    restored, restored_history = load_training(config)
    assert restored_history == history
    np.testing.assert_allclose(model(images[:1]), restored(images[:1]))
