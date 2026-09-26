# Testes unitários
import pytest
import tensorflow as tf
from app import models

def test_model_structure():
    """Verifica se o modelo é construído corretamente"""
    model = models.build_cnn(num_conv=2, filters=32, kernel_size=(3,3), activation="relu")
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 0

def test_model_output_shape_mnist():
    """Verifica se a saída do modelo para MNIST tem 10 classes"""
    model = models.build_cnn(num_conv=2, filters=32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1))
    output_shape = model.output_shape
    assert output_shape[-1] == 10

def test_model_output_shape_cifar():
    """Verifica se a saída do modelo para CIFAR-10 tem 10 classes"""
    model = models.build_cnn(num_conv=3, filters=64, kernel_size=(3,3), activation="relu", input_shape=(32,32,3))
    output_shape = model.output_shape
    assert output_shape[-1] == 10

def test_training_step():
    """Testa se o modelo consegue treinar por 1 época sem erro"""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:1000] / 255.0
    y_train = y_train[:1000]
    x_train = x_train.reshape(-1,28,28,1)

    model = models.build_cnn(num_conv=1, filters=16, kernel_size=(3,3), activation="relu", input_shape=(28,28,1))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)

    assert "accuracy" in history.history
