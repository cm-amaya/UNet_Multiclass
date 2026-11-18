import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from unet_segmentation.models import create_unet
from train.data_loader import Dataset, Dataloader
from train.utils import parse_args, ensure_dir


def main() -> None:
    """Entry point for UNet multiclass segmentation training."""
    # Parse args
    args = parse_args()
    if not args.get('positional'):
        parser = argparse.ArgumentParser(description='UNET for Multiclass Semantic Segmentation')
        parser.add_argument('train_dir', help='Train dir - with subdirs images and masks')
        parser.add_argument('val_dir', help='Val dir - with subdirs images and masks')
        parser.add_argument('--result_dir', default="results", help='Result dir')
        parser.add_argument('--image_size', type=int, default=320)
        parser.add_argument('--image_channels', type=int, default=3)
        parser.add_argument('--padding_size', type=int, default=800)
        parser.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=100)
        args = parser.parse_args()

    # Directories
    train_dir, val_dir = args.train_dir, args.val_dir
    result_dir = args.result_dir
    ensure_dir(result_dir)

    x_train_dir = os.path.join(train_dir, 'images')
    y_train_dir = os.path.join(train_dir, 'masks')
    x_valid_dir = os.path.join(val_dir, 'images')
    y_valid_dir = os.path.join(val_dir, 'masks')

    # Parameters
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = args.batch_size
    LR = 0.0001
    EPOCHS = args.epochs
    activation = 'softmax'
    n_classes = args.n_classes
    image_size = args.image_size
    image_channels = args.image_channels
    padding_size = args.padding_size

    # Datasets
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=n_classes, augment=True)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=n_classes, augment=False)

    # Model
    model = create_unet(BACKBONE, classes=n_classes, activation=activation)
    optim = keras.optimizers.Adam(LR)
    model.compile(
        optimizer=optim,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)

    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty. Check {x_train_dir} and {y_train_dir}")
    if len(valid_dataset) == 0:
        raise ValueError(f"Validation dataset is empty. Check {x_valid_dir} and {y_valid_dir}")

    batch = next(iter(train_dataloader))
    assert batch[0].shape == (BATCH_SIZE, image_size, image_size, image_channels), \
        f"Expected batch shape ({BATCH_SIZE}, {image_size}, {image_size}, {image_channels})"
    assert batch[1].shape == (BATCH_SIZE, image_size, image_size, n_classes), \
        f"Expected label shape ({BATCH_SIZE}, {image_size}, {image_size}, {n_classes})"

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
        epochs=EPOCHS,
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
