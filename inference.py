import argparse
import os
import numpy as np
import torch
import nibabel as nib
import SimpleITK as sitk
from unet3d import UNet3DDeep

import scipy.ndimage as ndi

def post_process_mask(mask_xyz, min_hole_size=500, closing_radius=1):
    """
    Fill holes and remove small spurious objects.
    mask_xyz: binary array in (x, y, z) orientation (same as nibabel output).
    """
    # Binary closing to bridge thin gaps
    struct = ndi.generate_binary_structure(3, 2)
    closed = ndi.binary_closing(mask_xyz, structure=struct, iterations=closing_radius)
    # Fill internal holes
    filled = ndi.binary_fill_holes(closed)
    # Remove small objects (likely false positives)
    labeled, num_features = ndi.label(filled)
    sizes = ndi.sum(filled, labeled, range(1, num_features + 1))
    mask_clean = np.isin(labeled, np.where(sizes >= min_hole_size)[0] + 1)
    return mask_clean.astype(np.uint8)


def n4_bias_correction(image_path: str):
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    mask  = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(image, mask)


def load_model(ckpt_path: str, device: torch.device):
    model = UNet3DDeep(
        in_channels=1, num_classes=1,
        level_channels=[32, 64, 128, 256],
        bottleneck_channel=512,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def _normalize_nonzero(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = vol != 0
    if not mask.any():
        return vol
    mean = vol[mask].mean()
    std  = vol[mask].std()
    return (vol - mean) / (std + eps)


def _save_like(base_img: nib.Nifti1Image, data: np.ndarray, path: str, dtype=np.float32):
    """
    Save data in the same world space as base_img.

    We do NOT copy the original header — that carries stale dim/pixdim fields
    that describe the original voxel layout and would conflict with the affine,
    causing viewers to misplace the mask. Let nibabel build a clean header from
    the data shape + affine, then only transplant the qform/sform codes.
    """
    img = nib.Nifti1Image(data.astype(dtype), base_img.affine)
    qform, qcode = base_img.get_qform(coded=True)
    sform, scode = base_img.get_sform(coded=True)
    if qform is not None:
        img.set_qform(qform, int(qcode))
    if sform is not None:
        img.set_sform(sform, int(scode))
    nib.save(img, path)


def run_inference(
    input_path: str,
    checkpoint_path: str,
    output_path: str | None,
    threshold: float = 0.5,
    use_bias_correction: bool = False,
    save_mask: bool = True,
    stripped_output: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(checkpoint_path, device)

    # ------------------------------------------------------------------ #
    # Load — nibabel gives (x, y, z) = (R-L, A-P, S-I)                  #
    # ------------------------------------------------------------------ #
    img_nib  = nib.load(input_path)
    orig_xyz = img_nib.get_fdata().astype(np.float32)   # (x, y, z)

    if use_bias_correction:
        # SimpleITK: GetArrayFromImage → (z, y, x); transpose to (x, y, z)
        img_sitk = n4_bias_correction(input_path)
        sitk_arr = sitk.GetArrayFromImage(img_sitk)         # (z, y, x)
        orig_xyz = np.transpose(sitk_arr, (2, 1, 0))        # (x, y, z)

    # ------------------------------------------------------------------ #
    # Pre-process — mirrors dataset.py exactly                            #
    #   dataset.py:  np.moveaxis(image, 1, 0)  →  (x,y,z) → (y,x,z)    #
    #                                               A-P axis becomes D    #
    # ------------------------------------------------------------------ #
    img_np = np.moveaxis(orig_xyz, 1, 0)    # (x,y,z) → (y,x,z)
    img_np = _normalize_nonzero(img_np)

    # Build tensor (1, 1, D, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        output    = model(img_tensor.to(device))     # eval → single tensor
        output    = torch.sigmoid(output)
        output    = (output > threshold).float()
        output_np = output.squeeze().cpu().numpy()   # (y, x, z)

    output_np = np.clip(output_np, 0, 1)

    # Invert: (y,x,z) → (x,y,z)  — move axis 0 back to position 1
    mask_xyz = np.moveaxis(output_np, 0, 1)

    mask_xyz = post_process_mask(mask_xyz, min_hole_size=500, closing_radius=1)

    # Sanity check
    assert mask_xyz.shape == orig_xyz.shape, (
        f"Shape mismatch after axis inversion: "
        f"mask {mask_xyz.shape} vs image {orig_xyz.shape}"
    )

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #
    if save_mask and output_path:
        _save_like(img_nib, mask_xyz, output_path, dtype=np.uint8)
        print(f"Mask saved → {output_path}")

    if stripped_output:
        stripped = orig_xyz * mask_xyz              # both (x, y, z)
        _save_like(img_nib, stripped, stripped_output, dtype=np.float32)
        print(f"Stripped volume saved → {stripped_output}")


def parse_args():
    p = argparse.ArgumentParser(description="3D skull stripping inference")
    p.add_argument("--input",              required=True,
                   help="Input NIfTI path (.nii or .nii.gz)")
    p.add_argument("--checkpoint",         required=True,
                   help="Model checkpoint path (.pth)")
    p.add_argument("--output",
                   help="Output mask path (.nii.gz)")
    p.add_argument("--stripped_output",
                   help="Skull-stripped volume output path (.nii.gz)")
    p.add_argument("--no_bias_correction", action="store_true",
                   help="Skip N4 bias field correction")
    p.add_argument("--no_mask",            action="store_true",
                   help="Do not save the binary mask")
    p.add_argument("--threshold",          type=float, default=0.5,
                   help="Sigmoid threshold for binarisation (default: 0.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        use_bias_correction=not args.no_bias_correction,
        save_mask=not args.no_mask,
        stripped_output=args.stripped_output,
    )