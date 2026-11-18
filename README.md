# UNet for Multiclass Semantic Segmentation

UNet for Multiclass Semantic Segmentation, using Keras 3.x with TensorFlow 2.x

## Requirements

You must have Python 3.10+ and an NVIDIA GPU with CUDA 11+ support (optional, for GPU acceleration).

### Installation using uv (Recommended)

uv is a fast Python package installer and resolver. Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

This will create a virtual environment at `.venv` and install all required packages.

### Installation using pip

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── unet_segmentation/
│   │   ├── __init__.py   # Package exports
│   │   └── models.py     # U-Net model definition
│   └── train/
│       ├── __init__.py   # Training orchestration
│       ├── data_loader.py    # Dataset, Dataloader classes
│       ├── augmentation.py   # Albumentations transforms
│       └── utils.py          # Helper utilities
├── main.py              # CLI entry point
├── pyproject.toml       # uv/pip dependency configuration
├── MANIFEST.in          # Package distribution manifest
├── LICENSE              # MIT License
└── README.md
```

## How to Use

### Training

Positional arguments are required (train_dir, val_dir). Additional options via `--`:

```bash
uv run python main.py \
    data/train \
    data/val \
    --n_classes 2 \
    --epochs 100
```

Or use the CLI entry point:

```bash
uv run train-unet data/train data/val --n_classes 2 --epochs 100
```

### Dataset Format

- `train_dir/images/` - Training images
- `train_dir/masks/` - Training masks (same filenames)
- Same structure for validation directory

### Arguments

| Argument | Description |
|----------|-------------|
| `train_dir` | Path to training folder (must contain `images/` and `masks/` subdirs) |
| `val_dir` | Path to validation folder (must contain `images/` and `masks/` subdirs) |
| `--result_dir` | Output directory for saved models and plots (default: `results`) |
| `--image_size` | Input image size for training (default: `320`) |
| `--image_channels` | Number of input image channels (default: `3`) |
| `--padding_size` | Padding size for validation images (default: `800`) |
| `--n_classes` | Number of segmentation classes (default: `2`) |
| `--batch_size` | Batch size for training (default: `2`) |
| `--epochs` | Number of training epochs (default: `100`) |

## Notes

- Uses Keras 3.x with TensorFlow 2.18+ for better maintenance and performance
- Models saved in TensorFlow SavedModel format (modern format, supports TF 2.x/3.x)
- Training plots saved as PNG in results directory
- CPU-only training supported; GPU acceleration via CUDA if available
- Albumentations for data augmentation (horizontal flip, rotation, crop, etc.)
