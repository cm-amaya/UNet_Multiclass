"""Tests for augmentation transforms."""
import numpy as np
import pytest
from train.augmentation import get_transforms


class TestGetTransforms:
    """Test transform creation."""

    def test_train_transforms(self):
        """Test training transforms are created."""
        transform = get_transforms(augment=True)

        assert transform is not None

    def test_val_transforms(self):
        """Test validation transforms are created."""
        transform = get_transforms(augment=False)

        assert transform is not None

    def test_transform_output_shape(self):
        """Test transform preserves shape."""
        transform = get_transforms(augment=True)

        img = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (32, 32)).astype(np.float32)

        transformed = transform(image=img, mask=mask)

        assert transformed['image'].shape == img.shape
        assert transformed['mask'].shape == mask.shape

    def test_transform_deterministic(self):
        """Test transforms are deterministic with seed."""
        transform = get_transforms(augment=True)

        img = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (32, 32)).astype(np.float32)

        result1 = transform(image=img.copy(), mask=mask.copy())
        result2 = transform(image=img.copy(), mask=mask.copy())

        assert np.allclose(result1['image'], result2['image'])
        assert np.allclose(result1['mask'], result2['mask'])

    def test_transform_val_no_augmentation(self):
        """Test validation transforms don't augment."""
        transform = get_transforms(augment=False)

        img = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (32, 32)).astype(np.float32)

        result = transform(image=img, mask=mask)

        # Should be identical (or very close due to float precision)
        assert np.allclose(result['image'], img)
        assert np.allclose(result['mask'], mask)