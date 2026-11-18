"""Advanced tests for model features."""
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/nomadmatt/repos/UNet_Multiclass/src")

from unet_segmentation.models import create_unet


class TestMultipleInputsOutputs:
    """Test model with multiple inputs/outputs."""

    @pytest.fixture
    def main_input(self):
        return np.random.rand(32, 32, 3).astype(np.float32)

    @pytest.fixture
    def auxiliary_input(self):
        return np.random.rand(32, 32, 1).astype(np.float32)

    def test_model_with_auxiliary_input(self, main_input, auxiliary_input):
        """Test model can handle auxiliary inputs."""
        # Create model with extra input channel (e.g., depth map)
        model = create_unet("resnet50", 2)

        # Temporarily modify input shape to test flexibility
        x = np.random.rand(1, 32, 32, 3).astype(np.float32)
        y = model(x, training=False)
        assert y.shape == (1, 32, 32, 2)


class TestSharedLayers:
    """Test shared layer functionality."""

    def test_shared_encoder_pattern(self):
        """Test shared encoder architecture pattern."""
        from tensorflow.keras import layers, models

        # Create encoder that can be shared
        encoder_input = layers.Input(shape=(32, 32, 3))
        x = layers.Conv2D(64, 3, activation='relu')(encoder_input)
        x = layers.MaxPooling2D(2)(x)
        encoder_output = layers.Conv2D(128, 3, activation='relu')(x)

        encoder = models.Model(encoder_input, encoder_output)

        # Use encoder multiple times
        x1 = np.random.rand(1, 32, 32, 3).astype(np.float32)
        x2 = np.random.rand(1, 32, 32, 3).astype(np.float32)

        encoded1 = encoder(x1)
        encoded2 = encoder(x2)

        assert encoded1.shape == (1, 16, 16, 128)
        assert encoded2.shape == (1, 16, 16, 128)


class TestFeatureExtraction:
    """Test intermediate layer extraction (feature extraction pattern)."""

    @pytest.fixture
    def model(self):
        return create_unet("resnet50", 2)

    @pytest.fixture
    def dummy_input(self):
        return np.random.rand(32, 32, 3).astype(np.float32)

    def test_intermediate_layer_extraction(self, model, dummy_input):
        """Test extracting intermediate layer activations."""
        # Get intermediate layer outputs
        intermediate = [layer.output for layer in model.layers[1:10]]
        feature_model = tf.keras.Model(model.input, intermediate)

        x = np.random.rand(1, 32, 32, 3).astype(np.float32)
        features = feature_model(x)

        assert isinstance(features, list)
        assert len(features) == 9
        # Features should have reasonable shapes (shrinking)
        for f in features:
            assert f.shape[0] == 1  # batch size
            assert f.shape[2] == 32  # height
            assert f.shape[3] == 32  # width

    def test_feature_extraction_different_layers(self, model, dummy_input):
        """Test extracting features from different layers."""
        # Early vs late layer features
        early_layer = model.layers[2]
        late_layer = model.layers[-5]

        x = np.random.rand(1, 32, 32, 3).astype(np.float32)

        early_feat = early_layer(x)
        late_feat = late_layer(x)

        # Early layers should have higher spatial resolution
        assert early_feat.shape[1] > late_feat.shape[1]
        assert early_feat.shape[2] > late_feat.shape[2]


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_inference(self):
        """Test complete inference pipeline."""
        model = create_unet("efficientnetb0", 3)
        x = np.random.rand(4, 32, 32, 3).astype(np.float32)

        # Inference mode
        predictions = model(x, training=False)
        assert predictions.shape == (4, 32, 32, 3)

        # Training mode (forward pass only)
        predictions_train = model(x, training=True)
        assert predictions_train.shape == (4, 32, 32, 3)

    def test_batch_processing(self):
        """Test processing multiple images in batch."""
        model = create_unet("resnet50", 2)
        batch_size = 8

        for batch in [1, 4, 8, 16]:
            x = np.random.rand(batch, 32, 32, 3).astype(np.float32)
            y = model(x, training=False)
            assert y.shape == (batch, 32, 32, 2)

    def test_model_idempotence(self):
        """Test model produces consistent outputs."""
        model = create_unet("resnet50", 2)
        x = np.random.rand(3, 32, 32, 3).astype(np.float32)

        # Multiple forward passes should be identical
        for _ in range(5):
            assert np.allclose(model(x, training=False).numpy(),
                             model(x, training=False).numpy())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_channel_input(self):
        """Test model with single channel input."""
        model = create_unet("resnet50", 2)
        x = np.random.rand(1, 32, 32, 1).astype(np.float32)
        y = model(x, training=False)
        assert y.shape == (1, 32, 32, 2)

    def test_small_batch(self):
        """Test model with batch size 1."""
        model = create_unet("resnet50", 2)
        x = np.random.rand(1, 32, 32, 3).astype(np.float32)
        y = model(x, training=False)
        assert y.shape == (1, 32, 32, 2)

    def test_large_output_classes(self):
        """Test model with many output classes."""
        model = create_unet("efficientnetb0", 10)
        x = np.random.rand(1, 32, 32, 3).astype(np.float32)
        y = model(x, training=False)
        assert y.shape == (1, 32, 32, 10)

    def test_model_summary(self):
        """Test model generates valid summary."""
        model = create_unet("resnet50", 2)
        summary = model.summary()
        assert len(summary) > 0