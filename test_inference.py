import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95, dc, assd
from config_test import (
    TEST_BATCH_SIZE, TEST_CUDA, DATASET_PATH, NUM_CLASSES,
    BACKGROUND_AS_CLASS
)
from transforms import val_transform, val_transform_cuda
from dataset import SkullStrippingDataset
from torch.utils.data import DataLoader
from unet3d import UNet3D

# Create output directories if they don't exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/predictions", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() and TEST_CUDA else "cpu")
print(f"Using device: {device}")

# Load model
checkpoint_path = "./checkpoints_16_32_64_128/epoch026_valLoss0.088994.pth"
if not os.path.exists(checkpoint_path):
    # Find the most recent checkpoint if checkpoint.pth doesn't exist
    checkpoint_files = [f for f in os.listdir("./checkpoints") if f.endswith('.pth')]
    if checkpoint_files:
        checkpoint_files.sort()  # Sort alphabetically
        checkpoint_path = os.path.join("./checkpoints", checkpoint_files[-1])
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError("No checkpoint found in ./checkpoints directory")

# Instantiate model
model = UNet3D(in_channels=1, num_classes=1, level_channels=[16, 32, 64], bottleneck_channel=128)
model.to(device)

# Load model weights
print(f"Loading model weights from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Set transforms based on device
test_transforms = val_transform_cuda if torch.cuda.is_available() and TEST_CUDA else val_transform

# Get test dataloader (avoid requiring train/valid folders)
test_dataset = SkullStrippingDataset(DATASET_PATH, mode='test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# Evaluation metrics
def calculate_metrics(pred, target):
    """Calculate evaluation metrics."""
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    
    # Binarize prediction using threshold 0.5
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    
    # Skip empty masks (avoiding division by zero in metrics)
    if np.sum(target_np) == 0 and np.sum(pred_binary) == 0:
        return {
            "dice": 1.0,  # Both are empty, perfect match
            "hd95": 0.0,
            "assd": 0.0,
        }
    elif np.sum(target_np) == 0 or np.sum(pred_binary) == 0:
        return {
            "dice": 0.0,  # One is empty, one is not
            "hd95": float('inf'),
            "assd": float('inf'),
        }
    
    # Calculate metrics
    try:
        dice_score = dc(pred_binary, target_np)
        hausdorff_distance = hd95(pred_binary, target_np)
        average_surface_distance = assd(pred_binary, target_np)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "dice": np.nan,
            "hd95": np.nan,
            "assd": np.nan,
        }
    
    return {
        "dice": dice_score,
        "hd95": hausdorff_distance,
        "assd": average_surface_distance,
    }

# Create a visualization function to save slices as images
def visualize_slices(image, mask, prediction, sample_name, slice_axis=0, num_slices=8):
    """
    Save visualization of slices from volume.
    
    Args:
        image: Input image volume
        mask: Ground truth mask volume
        prediction: Predicted mask volume
        sample_name: Name for saving the visualization
        slice_axis: Axis along which to take slices (0=depth, 1=height, 2=width)
        num_slices: Number of slices to visualize
    """
    # Get rid of batch and channel dimensions
    image = image.squeeze()
    mask = mask.squeeze()
    prediction = prediction.squeeze()
    
    # Determine total number of slices along the chosen axis
    total_slices = image.shape[slice_axis]
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 2, 6))
    
    for i, slice_idx in enumerate(slice_indices):
        # Extract slices
        if slice_axis == 0:
            img_slice = image[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
        elif slice_axis == 1:
            img_slice = image[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
        else:  # slice_axis == 2
            img_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
        
        # Plot images
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].set_title(f"Slice {slice_idx}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_slice, cmap='viridis')
        axes[1, i].set_title(f"GT Mask")
        axes[1, i].axis('off')
        
        axes[2, i].imshow(pred_slice, cmap='viridis')
        axes[2, i].set_title(f"Prediction")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    fig.savefig(f"results/visualizations/{sample_name}.png", dpi=300)
    plt.close(fig)

# Run inference
all_metrics = []
print("Starting inference on test dataset...")

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Extract data
        image, mask = batch["image"], batch["label"]
        sample_name = batch["name"][0]  # Get the name of the sample
        # Move data to device
        if torch.cuda.is_available() and TEST_CUDA:
            image = image.cuda()
            mask = mask.cuda()
        
        # Forward pass
        output = model(image)
        
        # Apply sigmoid to get probabilities
        prediction = torch.sigmoid(output)
        
        # Calculate metrics
        metrics = calculate_metrics(prediction, mask)
        metrics["sample_name"] = sample_name
        all_metrics.append(metrics)
        
        # Save prediction as NIfTI file
        prediction_np = prediction.cpu().numpy().squeeze()
        prediction_binary = (prediction_np > 0.5).astype(np.uint8)
        
        # Get original shape and convert back to the original dimension order
        image_np = image.cpu().numpy().squeeze()
        mask_np = mask.cpu().numpy().squeeze()
        
        # Save visualization
        visualize_slices(image_np, mask_np, prediction_binary, sample_name)
        
        # Save as Nifti (if desired)
        # Create a nibabel image object and save
        # pred_nifti = nib.Nifti1Image(prediction_binary, np.eye(4))
        # nib.save(pred_nifti, f"results/predictions/{sample_name}_pred.nii.gz")
        
        # print(f"Processed sample {sample_name}: Dice={metrics['dice']:.4f}, HD95={metrics['hd95']:.4f}, ASSD={metrics['assd']:.4f}")

        image_dir = os.path.join(DATASET_PATH, 'test', 'images')
        img_path = os.path.join(image_dir, sample_name)
        original_nifti = nib.load(img_path)
        
        # Prepare prediction array with the correct orientation
        # Need to move axes from (D, H, W) back to (W, H, D) if needed
        if len(prediction_binary.shape) == 3:  # If 3D volume
            # Transpose to match the original NIfTI orientation
            # This might need adjustment based on your dataset's specific orientation
            prediction_binary_reoriented = np.moveaxis(prediction_binary, 0, -1)
            
            # Ensure dimensions match the original
            if prediction_binary_reoriented.shape != original_nifti.shape:
                print(f"Warning: Shape mismatch - Original: {original_nifti.shape}, Prediction: {prediction_binary_reoriented.shape}")
                # Resize if necessary (this is a simple example, might need more sophisticated approach)
                if len(original_nifti.shape) == 3:  # If 3D volume
                    # Use the original shape
                    target_shape = original_nifti.shape
                    # Create a new array of zeros with the target shape
                    resized_pred = np.zeros(target_shape, dtype=np.uint8)
                    # Copy the prediction into the resized array (up to the minimum dimensions)
                    min_x = min(target_shape[0], prediction_binary_reoriented.shape[0])
                    min_y = min(target_shape[1], prediction_binary_reoriented.shape[1])
                    min_z = min(target_shape[2], prediction_binary_reoriented.shape[2])
                    resized_pred[:min_x, :min_y, :min_z] = prediction_binary_reoriented[:min_x, :min_y, :min_z]
                    prediction_binary_reoriented = resized_pred
        else:
            prediction_binary_reoriented = prediction_binary
        
        # Create a new NIfTI image with the original affine and header
        pred_nifti = nib.Nifti1Image(prediction_binary_reoriented, 
                                     original_nifti.affine, 
                                     header=original_nifti.header)
        
        # Save the NIfTI file
        nib.save(pred_nifti, f"results/predictions/{sample_name}_pred.nii.gz")
        
        print(f"Processed sample {sample_name}: Dice={metrics['dice']:.4f}, HD95={metrics['hd95']:.4f}, ASSD={metrics['assd']:.4f}")

# Calculate and print average metrics
avg_metrics = {
    "dice": np.mean([m["dice"] for m in all_metrics]),
    "hd95": np.mean([m["hd95"] for m in all_metrics]),
    "assd": np.mean([m["assd"] for m in all_metrics]),
}

print("\n\n====== Test Results ======")
print(f"Average Dice: {avg_metrics['dice']:.4f}")
print(f"Average HD95: {avg_metrics['hd95']:.4f}")
print(f"Average ASSD: {avg_metrics['assd']:.4f}")

# Save metrics to CSV
import csv
with open("results/test_metrics.csv", "w", newline="") as csvfile:
    fieldnames = ["sample_name", "dice", "hd95", "assd"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for metric in all_metrics:
        writer.writerow(metric)

print(f"Results saved to 'results' directory")
