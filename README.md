# UNet for Multiclass Semantic Segmentation

UNet for Multiclass Semantic Segmentation, on Keras, based on Segmentation Models' Unet libray

## Requirements

You must have Python 3 and an NVIDIA GPU with CUDA 10 support, by means of the pip tool you can install the packages using the following command on the repository folder:

```code
pip install -r requirements.txt
```

Some important libraries:

* keras >= 2.2.0 or tensorflow >= 1.13
* albumentations==0.3.0
* segmentation-models==1.0.*

**Install Segmentation Models**

```code
 pip install git+https://github.com/qubvel/segmentation_models
```

## Project Structure

    .
    ├──augmentation.py: Module with functions for image sets augmentation.
    ├──data_loader.py: Module with the class for the manipulation of the image sets
    ├──main.py: Main module of the code, receives several arguments for training or testing models.
    ├──utils.py: Module with auxiliary functions.
    └──requirements.txt: File containing a list of the libraries required to run all the modules.

## How to Use

A call must be made to the main module, passing it the arguments corresponding to the dataset paths and the training parameters.

**Arguments**

```code
-h, --help: Show main module arguments.
--train_dir: Path where the training folder is, it must contain the images and masks folders.
--val_dir: Path where the validation folder is, it must contain the images and masks folders.
--result_dir: Path where the resulting models are saved.
--image_size: Standard size for all input images, crop if necessary.
--image_channels: Number of channels of the input images
--padding_size: Padding size for Val images - must be a multiple of image size.
--n_classes: Number of classes in the masks.
--batch_size: Batch size, to be adjusted depending on GPU capacity.
--epochs: Number of training epochs.
```
