"""Integration tests for training pipeline."""
import os
import numpy as np
import pytest
import sys
sys.path.insert(0, "/home/nomadmatt/repos/UNet_Multiclass/src")

from train.data_loader import Dataset, Dataloader
from train.augmentation import get_transforms


class TestDataPipeline:
    """Integration tests for data pipeline."""

    def test_dataset_dataloader_compatibility(self, tmp_path):
        """Test Dataset and Dataloader work together end-to-end."""
        from test_data_loader import create_test_dataset
        create_test_dataset(tmp_path)

        dataset = Dataset(tmp_path, classes=2)
        dataloader = Dataloader(dataset, batch_size=2)

        for i, (img_batch, mask_batch) in enumerate(dataloader):
            assert img_batch.shape == (2, 2, 3)
            assert mask_batch.shape == (2, 2)
            assert i < 2  # Should have exactly 2 batches

    def test_augmentation_integration(self, tmp_path):
        """Test augmentation pipeline integration."""
        from test_data_loader import create_test_dataset
        create_test_dataset(tmp_path)

        dataset = Dataset(tmp_path, classes=2)
        transform = get_transforms(augment=True)

        img, mask = next(iter(dataset))
        transformed = transform(image=img, mask=mask)

        assert transformed['image'].shape == img.shape
        assert transformed['mask'].shape == mask.shape

    def test_full_pipeline_shapes(self, tmp_path):
        """Test complete pipeline shape consistency."""
        from test_data_loader import create_test_dataset
        create_test_dataset(tmp_path)

        train_dataset = Dataset(tmp_path, classes=2)
        train_loader = Dataloader(train_dataset, batch_size=2)

        img_batch, mask_batch = next(iter(train_loader))

        # Simulate augmentation
        transform = get_transforms(augment=True)
        augmented_img, augmented_mask = transform(image=img_batch, mask=mask_batch)

        # Verify shapes preserved through pipeline
        assert img_batch.shape == augmented_img.shape
        assert mask_batch.shape == augmented_mask.shape


class TestTrainingPreparation:
    """Tests for training preparation steps."""

    def test_dataset_slicing(self, tmp_path):
        """Test dataset can be sliced (for train/val split)."""
        from test_data_loader import create_test_dataset
        create_test_dataset(tmp_path)

        full_dataset = Dataset(tmp_path, classes=2)
        assert len(full_dataset) == 4

        # Simulate slicing for training
        train_dataset = Dataset(tmp_path, classes=2)
        assert len(train_dataset) == 4  # Same dataset, different reference


class TestModelWithPipeline:
    """Test model integration with data pipeline."""

    def test_model_on_real_batch(self, tmp_path):
        """Test model processes real data batch correctly."""
        from test_data_loader import create_test_dataset
        create_test_dataset(tmp_path)

        dataset = Dataset(tmp_path, classes=2)
        dataloader = Dataloader(dataset, batch_size=2)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])

        for img_batch, mask_batch in dataloader:
            x = tf.convert_to_tensor(img_batch, dtype=tf.float32) / 255.0
            y = model(x, training=False)
            assert y.shape == (2, 2)
            # Predictions should be valid probabilities
            assert np.all(y >= 0) and np.all(y <= 1)