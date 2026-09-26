"""Dataset metadata and explicit, on-demand loading."""

import numpy as np
from tensorflow import keras

DATASETS = {
    "MNIST": {"shape": (28, 28, 1), "classes": [str(i) for i in range(10)]},
    "FashionMNIST": {"shape": (28, 28, 1), "classes": [
        "Camiseta", "Calça", "Pulôver", "Vestido", "Casaco", "Sandália",
        "Camisa", "Tênis", "Bolsa", "Bota"]},
    "CIFAR-10": {"shape": (32, 32, 3), "classes": [
        "Avião", "Automóvel", "Pássaro", "Gato", "Cervo", "Cachorro",
        "Sapo", "Cavalo", "Navio", "Caminhão"]},
}


def load_dataset(name):
    loader = {"MNIST": keras.datasets.mnist, "FashionMNIST": keras.datasets.fashion_mnist,
              "CIFAR-10": keras.datasets.cifar10}[name]
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    shape = DATASETS[name]["shape"]
    return (
        x_train.reshape((-1, *shape)).astype(np.float32) / 255.0,
        y_train.reshape(-1),
        x_test.reshape((-1, *shape)).astype(np.float32) / 255.0,
        y_test.reshape(-1),
    )
