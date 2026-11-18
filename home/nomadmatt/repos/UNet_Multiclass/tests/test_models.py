"""Tests for UNet model creation."""
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K

# Import from installed package to avoid local path issues
import sys
sys.path.insert(0, "/home/nomadmatt/repos/UNet_Multiclass/src")

from unet_segmentation.models import create_unet


class TestCreateUnet:
    """Test UNet model creation."""

    @pytest.fixture
    def dummy_input(self):
        """Create a dummy input tensor."""
        return np.random.rand(32, 32, 3).astype(np.float32)

    @pytest.fixture
    def dummy_output(self, dummy_input):
        """Create a dummy output tensor with same shape."""
        return dummy_input.copy()

    @pytest.mark.parametrize("backbone", ["resnet50", "efficientnetb0"])
    def test_model_creation(self, dummy_input):
        """Test that model can be created with valid backbones."""
        model = create_unet("resnet50", 2)
        assert model is not None

        # Test inference
        x = tf.convert_to_tensor(dummy_input)
        y = model(x, training=False)
        assert y.shape == (32, 32, 2)

    def test_model_creation_invalid_backbone(self):
        """Test that invalid backbone raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            create_unet("invalid_backbone", 2)

    def test_model_creation_invalid_classes(self):
        """Test that invalid classes raises ValueError."""
        with pytest.raises(ValueError, match="classes must be >= 2"):
            create_unet("resnet50", 1)

    def test_model_creation_activation(self):
        """Test model with softmax activation."""
        model = create_unet("resnet50", 3, activation="softmax")
        x = tf.convert_to_tensor(np.random.rand(1, 32, 32, 3).astype(np.float32))
        y = model(x, training=False)
        assert y.shape == (1, 32, 32, 3)

    def test_model_deterministic(self):
        """Test that model creation is deterministic."""
        model1 = create_unet("resnet50", 2)
        model2 = create_unet("resnet50", 2)
        x = tf.convert_to_tensor(np.random.rand(1, 32, 32, 3).astype(np.float32))
        assert np.allclose(model1(x).numpy(), model2(x).numpy())

    def test_model_output_dtype(self):
        """Test output dtype matches activation."""
        model = create_unet("resnet50", 2, activation="softmax")
        x = tf.convert_to_tensor(np.random.rand(1, 32, 32, 3).astype(np.float32))
        y = model(x, training=False)
        # Softmax outputs float32
        assert y.dtype == tf.float32

    def test_model_reuse(self):
        """Test model can be used multiple times."""
        model = create_unet("resnet50", 2)
        x = tf.convert_to_tensor(np.random.rand(4, 32, 32, 3).astype(np.float32))
        for _ in range(10):
            y = model(x, training=False)
            assert y.shape == (4, 32, 32, 2)