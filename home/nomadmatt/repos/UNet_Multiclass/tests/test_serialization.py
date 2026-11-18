"""Tests for model serialization and loading."""
import os
import tempfile
import numpy as np
import pytest

import tensorflow as tf
import sys
sys.path.insert(0, "/home/nomadmatt/repos/UNet_Multiclass/src")

from unet_segmentation.models import create_unet


class TestSerialization:
    """Test model save/load functionality."""

    @pytest.fixture
    def dummy_input(self):
        return np.random.rand(32, 32, 3).astype(np.float32)

    @pytest.fixture
    def model(self, dummy_input):
        model = create_unet("resnet50", 2)
        # Do a quick forward pass to initialize weights
        _ = model(tf.convert_to_tensor(dummy_input), training=False)
        return model

    def test_model_save_load(self, model, dummy_input):
        """Test model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.keras")
            model.save(path)

            loaded = tf.keras.models.load_model(path)

        # Verify loaded model produces same outputs
        x = tf.convert_to_tensor(dummy_input)
        assert np.allclose(model(x).numpy(), loaded(x).numpy())

    def test_model_save_load_without_weights(self, model, dummy_input):
        """Test model can be saved without weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model_no_weights.keras")
            model.save(path, weights=None)

            loaded = tf.keras.models.load_model(path)

        # Architecture should match
        assert len(model.layers) == len(loaded.layers)
        # Weights should be different (randomized)
        assert not np.allclose(model(x).numpy(), loaded(x).numpy())

    def test_model_save_load_with_weights(self, model, dummy_input):
        """Test model can be saved with weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model_with_weights.keras")
            model.save(path, weights="by_name")

            loaded = tf.keras.models.load_model(path)

        # Weights should be restored
        assert np.allclose(model(x).numpy(), loaded(x).numpy())

    def test_model_save_load_multiple_times(self, model, dummy_input):
        """Test model can be saved and loaded multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = os.path.join(tmpdir, f"model_{i}.keras")
                model.save(path)

                loaded = tf.keras.models.load_model(path)
                assert np.allclose(model(x).numpy(), loaded(x).numpy())

    def test_model_save_load_architecture_only(self, model):
        """Test architecture-only save preserves layer types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arch_only.keras")
            model.save(path, weights=None)

            loaded = tf.keras.models.load_model(path)

        # Verify same layer types
        for i, (orig, loaded_layer) in enumerate(zip(model.layers, loaded.layers)):
            assert type(orig).__name__ == type(loaded_layer).__name__

    def test_model_save_load_compatibility(self, model, dummy_input):
        """Test model loads correctly after full save cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "full_cycle.keras")
            model.save(path)

            # Delete reference to original model
            del model
            loaded = tf.keras.models.load_model(path)

        x = tf.convert_to_tensor(dummy_input)
        assert loaded(x).shape == (32, 32, 2)

    def test_model_save_load_batch_size_variants(self, model, dummy_input):
        """Test model works with different batch sizes after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "batch_variants.keras")
            model.save(path)

            loaded = tf.keras.models.load_model(path)

        # Test different batch sizes
        for batch in [1, 4, 8, 16]:
            x = tf.random.uniform((batch, 32, 32, 3), dtype=tf.float32)
            y = loaded(x, training=False)
            assert y.shape == (batch, 32, 32, 2)

    def test_model_save_load_training_config(self, dummy_input):
        """Test model preserves training config after save/load."""
        model = create_unet("resnet50", 2)
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        _ = model(tf.convert_to_tensor(dummy_input), training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.keras")
            model.save(path)

            loaded = tf.keras.models.load_model(path)

        # Compile again for metrics comparison
        loaded.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        x = tf.convert_to_tensor(dummy_input)
        assert np.allclose(model(x).numpy(), loaded(x).numpy())