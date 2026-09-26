"""CNN construction shared by the interface and numerical tests."""

from tensorflow.keras import layers, models


def validate_architecture(num_conv, kernel_size, input_shape):
    height, width = input_shape[:2]
    for index in range(num_conv):
        height = (height - kernel_size[0] + 1) // 2
        width = (width - kernel_size[1] + 1) // 2
        if min(height, width) < 1:
            raise ValueError(
                f"O bloco {index + 1} reduz a imagem a uma dimensão inválida. "
                "Diminua o número de camadas ou o tamanho do kernel."
            )


def build_cnn(num_conv=2, filters=32, kernel_size=(3, 3), activation="relu",
              input_shape=(28, 28, 1), pooling="MaxPooling2D", dense_units=64):
    validate_architecture(num_conv, kernel_size, input_shape)
    inputs = layers.Input(shape=input_shape, name="imagem")
    x = inputs
    pool = layers.MaxPooling2D if pooling == "MaxPooling2D" else layers.AveragePooling2D
    for index in range(num_conv):
        x = layers.Conv2D(filters, kernel_size, activation=activation,
                          name=f"conv_{index + 1}")(x)
        x = pool((2, 2), name=f"pool_{index + 1}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    logits = layers.Dense(10, name="classification_logits")(x)
    outputs = layers.Activation("softmax", name="probabilidades")(logits)
    return models.Model(inputs, outputs, name="cnn_visualizer")


def model_from_config(cfg):
    from app.datasets import DATASETS
    return build_cnn(num_conv=cfg["num_conv"], filters=cfg["filters"],
                     kernel_size=(cfg["kernel"], cfg["kernel"]),
                     activation=cfg["activation"], pooling=cfg["pooling"],
                     dense_units=cfg["dense_units"], input_shape=DATASETS[cfg["dataset"]]["shape"])
