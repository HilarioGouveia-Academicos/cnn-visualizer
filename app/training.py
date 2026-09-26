"""Training service; reports progress without depending on UI widgets."""
import tensorflow as tf
from app.models import model_from_config
from app.settings import training_allowed


def train_model(config, dataset_loader, on_epoch=None):
    if not training_allowed():
        raise PermissionError("Treinamento desabilitado nesta instalação.")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A semente deve ser um inteiro entre 0 e 4294967295.")
    tf.keras.utils.set_random_seed(seed)

    class Metrics(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.values = {key: [] for key in ("loss", "val_loss", "accuracy", "val_accuracy")}

        def on_epoch_end(self, epoch, logs=None):
            for key in self.values:
                self.values[key].append(float(logs[key]))
            if on_epoch is not None:
                on_epoch(epoch, self.values)

    x_train, y_train, _, _ = dataset_loader(config["dataset"])
    model = model_from_config(config)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    result = model.fit(x_train, y_train, epochs=config["epochs"],
                       batch_size=config["batch_size"], validation_split=0.15,
                       callbacks=[Metrics()], verbose=0)
    return model, {key: [float(v) for v in values] for key, values in result.history.items()}
