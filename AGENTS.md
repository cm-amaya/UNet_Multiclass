# UNet Multiclass - AI Agent Context & Changelog

## Quick Context

- **Goal**: Multiclass semantic segmentation using U-Net with EfficientNet backbones
- **Stack**: TensorFlow 2.x + Keras 3.x + Albumentations
- **Status**: ✅ Running with modern dependencies (see `pyproject.toml`)

## Dependency History

| Version | Date | Change |
|---------|------|--------|
| 2.9.3 | Original | Legacy TensorFlow/Keras 2.3 setup |
| 2.18.1 | Nov 2026 | Upgraded to modern TF/Keras 3.x via uv |

## Major Changes

### Nov 2026 - uv Migration & Keras 3.x Upgrade
- **Why**: Original `requirements.txt` had outdated, unmaintained versions. Keras 2.3 + segmentation-models 0.0.12 incompatible with modern Python.
- **What**:
  - Replaced `pip` with `uv` as package manager
  - Migrated to Keras 3.x (`tensorflow.keras`) API
  - Removed `segmentation-models` dependency (compatibility issues)
  - Rewrote model creation using native Keras 3.x U-Net builder
  - Updated model saving to SavedModel format (modern TF format)
- **Files Changed**: `main.py`, `utils.py`, `pyproject.toml`, `requirements.txt` (removed), `README.md`

### Nov 2026 - Package Structure Standardization
- **Why**: Follow Python packaging best practices for maintainability and distribution
- **What**:
  - Moved model code to `src/unet_segmentation/` package
  - Moved training orchestration to `src/train/` subpackage (`data_loader`, `augmentation`, `utils`)
  - Added proper `__init__.py` exports and versioning
  - Updated `main.py` to import from packages (no inline imports)
  - Added `.gitignore`, `MANIFEST.in`, `LICENSE`, and type checking config
- **Files Changed**: Created `src/unet_segmentation/`, `src/train/`, updated `main.py`, added build configs

## Architecture Notes

### Package Structure
```
src/
├── unet_segmentation/
│   ├── __init__.py   # Package exports (version, create_unet)
│   └── models.py     # U-Net model definition
└── train/
    ├── __init__.py   # Training orchestration
    ├── data_loader.py    # Dataset, Dataloader classes
    ├── augmentation.py   # Albumentations transforms
    └── utils.py          # Helper utilities
```

### Model Creation (`src/unet_segmentation/models.py`)
```python
def create_unet(backbone_name, classes, activation='softmax'):
    # Keras 3.x Input -> Backbone -> Concat -> Dense layers
    # Input: (320, 320, 3)
    # Output: (320, 320, n_classes) with softmax
```

### Data Pipeline
- `src/train/data_loader.py`: Custom `Dataset`, `Dataloader` classes
- `src/train/augmentation.py`: Albumentations transforms

### Training Loop (`main.py`)
- Loss: SparseCategoricalCrossentropy + CategoricalFocalLoss
- Optimizer: Adam (lr=0.0001)
- Metrics: SparseCategoricalAccuracy
- Callbacks: ModelCheckpoint (best weights), ReduceLROnPlateau

## Known Limitations
- No external CUDA verification in current setup
- Training plots use matplotlib (basic line charts)
- Validation uses batch_size=1 for memory efficiency

## Quick Commands
```bash
# Install deps
uv sync

# Train (positional args: train_dir val_dir)
uv run python main.py data/train data/val --n_classes 2

# CLI entry point
uv run train-unet data/train data/val

# Check installed packages
uv pip list
```

---

## 📝 Maintenance Rule
**After every major change** (dependency upgrade, API migration, architecture shift, or breaking change), update this file with:
1. Timestamp & commit message
2. What changed & why
3. Files affected
4. Any AI-agent-relevant notes (architecture, quirks, gotchas)

This keeps context fresh for future agents working on the project.

*Last updated: Nov 18, 2026*
