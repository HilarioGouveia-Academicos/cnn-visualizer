"""Architecture rendering, activations and Grad-CAM without UI side effects."""
import numpy as np
import tensorflow as tf
import visualkeras
from tensorflow.keras import layers


def architecture_image(model, view="Camadas", sizing="balanced", reversed_view=False, flat=False):
    colors = {
        layers.Conv2D: {"fill": "#6384F5", "outline": "#3556B6"},
        layers.MaxPooling2D: {"fill": "#48BAA2", "outline": "#257A68"},
        layers.AveragePooling2D: {"fill": "#48BAA2", "outline": "#257A68"},
        layers.Flatten: {"fill": "#E5AF58", "outline": "#A7782D"},
        layers.Dense: {"fill": "#A28BE0", "outline": "#7155AA"},
        layers.Activation: {"fill": "#E78897", "outline": "#AD5362"},
    }
    if view == "Grafo":
        return visualkeras.graph_view(model, color_map=colors)
    if view in ("Functional View", "LeNet View"):
        renderer_name = "functional_view" if view == "Functional View" else "lenet_view"
        renderer = getattr(visualkeras, renderer_name, None)
        if renderer is None:
            raise RuntimeError(
                "Atualize as dependências com python -m pip install -r requirements.txt "
                "e reinicie o Streamlit para habilitar esta representação."
            )
        if view == "Functional View":
            return renderer(model, color_map=colors)
        return renderer(model)
    # VisualKeras 0.2 expects the Keras 2 attribute, removed in Keras 3.
    # Supply it only during rendering; leave the model unchanged afterwards.
    adapted = []
    try:
        for layer in model.layers:
            if not hasattr(layer, "output_shape"):
                layer.output_shape = tuple(layer.output.shape)
                adapted.append(layer)
        return visualkeras.layered_view(
            model, options={
                "legend": True, "color_map": colors, "spacing": 30,
                "legend_text_spacing_offset": 0,
                "sizing_mode": sizing, "draw_reversed": reversed_view,
                "draw_volume": not flat, "background_fill": "white",
                "show_dimension": True, "relative_base_size": 1,
            },
        )
    finally:
        for layer in adapted:
            del layer.output_shape


def feature_maps(model, image, layer_name):
    extractor = tf.keras.Model(model.input, model.get_layer(layer_name).output)
    return np.asarray(extractor(image, training=False))[0]


def grad_cam(model, image, layer_name, class_index=None):
    """Use logits without changing the classifier's softmax activation."""
    extractor = tf.keras.Model(model.input, [model.get_layer(layer_name).output,
                                            model.get_layer("classification_logits").output])
    with tf.GradientTape() as tape:
        activations, logits = extractor(tf.convert_to_tensor(image), training=False)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        score = logits[:, class_index]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("Não foi possível calcular gradientes para esta camada.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(activations[0] * weights, axis=-1))
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return np.asarray(heatmap), class_index
