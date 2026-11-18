"""Tests for DataLoader and Dataset."""
import os
import numpy as np
import pytest
import tempfile
import shutil

from train.data_loader import Dataset, Dataloader


def create_test_dataset(tmp_path):
    """Create a minimal test dataset."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # Create 4 test images (2x2 for simplicity)
    for i in range(4):
        img_path = images_dir / f"img_{i}.png"
        mask_path = masks_dir / f"img_{i}.png"

        # Simple 2x2 grayscale images
        img = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        np.save(img_path, img)
        np.save(mask_path, mask)

    return images_dir, masks_dir


@pytest.fixture
def test_dataset(tmp_path):
    """Create a test dataset."""
    create_test_dataset(tmp_path)
    return tmp_path


class TestDataset:
    """Test Dataset class."""

    def test_dataset_initialization(self, test_dataset):
        """Test dataset initializes correctly."""
        dataset = Dataset(test_dataset, classes=2)
        assert dataset.root == test_dataset
        assert dataset.classes == 2

    def test_dataset_load_samples(self, test_dataset):
        """Test samples are loaded correctly."""
        dataset = Dataset(test_dataset, classes=2)
        samples = list(dataset)

        assert len(samples) == 4

    def test_dataset_empty_directory(self, tmp_path):
        """Test dataset handles empty directories."""
        dataset = Dataset(tmp_path, classes=2)
        assert len(dataset) == 0

    def test_dataset_missing_mask(self, test_dataset):
        """Test dataset skips images without masks."""
        # Remove mask for one image
        img_path = test_dataset / "images" / "img_0.png"
        mask_path = test_dataset / "masks" / "img_0.png"
        if mask_path.exists():
            mask_path.unlink()

        dataset = Dataset(test_dataset, classes=2)
        samples = list(dataset)
        assert len(samples) == 3  # One skipped

    def test_dataset_image_shape(self, test_dataset):
        """Test image shape is correct."""
        dataset = Dataset(test_dataset, classes=2)
        img, mask = next(iter(dataset))
        assert img.shape == (2, 2, 3)

    def test_dataset_mask_shape(self, test_dataset):
        """Test mask shape is correct."""
        dataset = Dataset(test_dataset, classes=2)
        img, mask = next(iter(dataset))
        assert mask.shape == (2, 2)


class TestDataloader:
    """Test Dataloader class."""

    def test_dataloader_basic(self, test_dataset):
        """Test basic dataloader iteration."""
        dataset = Dataset(test_dataset, classes=2)
        dataloader = Dataloader(dataset, batch_size=2)

        batches = list(dataloader)
        assert len(batches) == 2  # 4 samples / 2 batch_size

    def test_dataloader_batch_shape(self, test_dataset):
        """Test batch shape is correct."""
        dataset = Dataset(test_dataset, classes=2)
        dataloader = Dataloader(dataset, batch_size=2)

        img_batch, mask_batch = next(iter(dataloader))
        assert img_batch.shape == (2, 2, 3)
        assert mask_batch.shape == (2, 2)

    def test_dataloader_shuffle(self, test_dataset):
        """Test shuffle works correctly."""
        dataset = Dataset(test_dataset, classes=2)

        # Without shuffle
        dataloader1 = Dataloader(dataset, batch_size=2, shuffle=False)
        batches1 = list(dataloader1)

        # With shuffle
        dataloader2 = Dataloader(dataset, batch_size=2, shuffle=True)
        batches2 = list(dataloader2)

        # Batches should be different (unless random seed is same)
        assert batches1 != batches2

    def test_dataloader_single_sample(self, tmp_path):
        """Test dataloader with single sample."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        img = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        np.save(images_dir / "img.png", img)
        np.save(masks_dir / "img.png", mask)

        dataset = Dataset(tmp_path, classes=2)
        dataloader = Dataloader(dataset, batch_size=1)

        img_batch, mask_batch = next(iter(dataloader))
        assert img_batch.shape == (1, 2, 3)
        assert mask_batch.shape == (1, 2)

    def test_dataloader_large_batch(self, test_dataset):
        """Test dataloader with batch_size >= dataset size."""
        dataset = Dataset(test_dataset, classes=2)
        dataloader = Dataloader(dataset, batch_size=100)

        img_batch, mask_batch = next(iter(dataloader))
        assert img_batch.shape == (4, 2, 3)  # All samples in one batch