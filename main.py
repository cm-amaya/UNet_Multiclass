

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from unet_segmentation.models import create_unet
from train.data_loader import Dataset, Dataloader
from train.utils import parse_args, ensure_dir


def load_config(path: str | None = None) -> dict:
    """Load optional config.json at repo root. Returns {} on any error (lenient)."""
    if not path:
        path = Path(__file__).parent / "config.json"
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    """Entry point for UNet multiclass segmentation training."""
    # Parse args
    args = parse_args()
    positional = args.get('positional', [])

    # Load config: positional[0] is custom config path
    cfg = load_config(positional[0] if len(positional) > 0 else None)

    # Apply config defaults, override with CLI args
    train_dir = Path(cfg.get("train_dir", positional[1] if len(positional) > 1 else "data/train"))
    val_dir = Path(cfg.get("val_dir", positional[2] if len(positional) > 2 else "data/val"))
    result_dir = Path(cfg.get("result_dir", "results"))
    image_size = cfg.get("image_size", 320)
    image_channels = cfg.get("image_channels", 3)
    padding_size = cfg.get("padding_size", 800)
    n_classes = cfg.get("n_classes", 2)
    batch_size = cfg.get("batch_size", 8)
    epochs = cfg.get("epochs", 50)
    backbone = cfg.get("backbone", "resnet50")
    lr = cfg.get("lr", 0.0001)

    # Directories
    x_train_dir = train_dir / "images"
    y_train_dir = train_dir / "masks"
    x_valid_dir = val_dir / "images"
    y_valid_dir = val_dir / "masks"
    ensure_dir(result_dir)

    # Datasets
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=n_classes, augment=True)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=n_classes, augment=False)

    # Model
    model = create_unet(backbone, classes=n_classes, activation='softmax')
    optim = keras.optimizers.Adam(lr)
    model.compile(
        optimizer=optim,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=batch_size, shuffle=False)

    assert train_dataloader[0][0].shape == (batch_size, image_size, image_size, image_channels)
    assert train_dataloader[0][1].shape == (batch_size, image_size, image_size, n_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(result_dir, 'best_model.h5'),
            save_weights_only=True,
            save_best_only=True,
            mode='min',
        ),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    history = model.fit(
        train_dataloader,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_dataloader,
    )

    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(result_dir, 'UNET-Training.png'))

    model.save(os.path.join(result_dir, 'best_model'))


if __name__ == '__main__':
    main()
