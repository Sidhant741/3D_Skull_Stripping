# 3D Skull Stripping (3D U-Net)

A 3D U-Net based skull-stripping pipeline for MRI volumes. This repo includes dataset loading, MONAI transforms, training, inference, and evaluation scripts for volumetric brain extraction.

**What this project does**
- Trains a 3D U-Net for binary brain mask prediction.
- Runs inference on NIfTI volumes and writes predicted masks.
- Evaluates Dice, HD95, and ASSD on a test split.

**Repo Structure**
- `train.py` Training script with TensorBoard logging and checkpointing.
- `test_inference.py` Evaluation on the test split with metrics and visualizations.
- `inference.py` Single-volume inference example.
- `dataset.py` Dataset and DataLoader definitions.
- `transforms.py` MONAI preprocessing and augmentation pipelines.
- `unet3d.py` 3D U-Net implementation.
- `config.py` Local training config (paths, batch sizes, etc.).
- `config_test.py` Test-time config.
- `config.sample.py` Template config (safe for GitHub).

**Setup**
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Data Layout**
Expected folder structure:

```
<DATASET_PATH>/
  train/
    images/
    masks/
  valid/
    images/
    masks/
  test/
    images/
    masks/
```

Mask naming assumes:
- `.nii.gz` images -> corresponding `_manual.nii.gz` masks
- `.nii` images -> corresponding `_ss.nii` masks

**Configuration**
Copy the template and update paths:

```bash
cp config.sample.py config.py
cp config.sample.py config_test.py
```

Then edit `DATASET_PATH`, batch sizes, and CUDA flags inside `config.py` and `config_test.py`.

**Training**
```bash
python train.py
```

**Inference (single volume)**
```bash
python inference.py
```

**Test Evaluation**
```bash
python test_inference.py
```

**Notes**
- This project uses the **Calgary Campinas Public Dataset**. The dataset is not included in this repository—please obtain it from the original source and comply with its license.
- Checkpoints and outputs are excluded from git by `.gitignore`.

**Example Outputs**
Add example results in `assets/` and reference them here if you want to showcase qualitative results.
