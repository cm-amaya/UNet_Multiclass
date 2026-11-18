"""U-Net model implementation using Keras 3.x APIs."""

import keras
from tensorflow.keras import layers


def create_unet(backbone_name: str, classes: int, activation: str = "softmax") -> "keras.Model":
    """Create a U-Net model using Keras 3.x APIs.

    Args:
        backbone_name: EfficientNet variant (e.g., 'efficientnetb3')
        classes: Number of segmentation classes
        activation: Output activation ('softmax' for multi-class, 'sigmoid' for binary)

    Returns:
        Compiled Keras model ready for training/inference
    """
    from keras.applications import get_model

    backbone = get_model(
        backbone_name.upper(),
        include_top=False,
        weights=None,
    )

    inputs = layers.Input(shape=(320, 320, 3))
    x = backbone(inputs)
    decoder_inputs = layers.Input(shape=(320, 320, classes))

    combined = layers.Concatenate()([x, decoder_inputs])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(combined)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(classes, 1, activation=activation)(x)

    return layers.Model(inputs, x)
