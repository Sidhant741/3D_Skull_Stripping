import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_to_output
from unet3d import UNet3D  
import os
import SimpleITK as sitk

def resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * ospc / nspc)) 
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)

def n4_bias_correction(image_path):
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)

    # Resample to (1,1,1) spacing
    image = resample_to_spacing(image, (1.0, 1.0, 1.0))
    
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, mask)
    return corrected_image

# Paths
input_path = "/storage/shagun/medical/dataset/test/images/CC0003_philips_15_63_F.nii.gz"
checkpoint_path = "./checkpoints/epoch003_valLoss0.703928.pth"
output_path = "output"


#img_nib = nib.load(input_path)
#img_resampled = resample_to_output(img_nib, voxel_sizes=(1, 1, 1))
#img_data = img_resampled.get_fdata()
#original_shape = img_data.shape

img_data = n4_bias_correction(input_path)
img_np = sitk.GetArrayFromImage(img_data)


#img_norm = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
#img_tensor = img_tensor.permute(0, 1, 4, 2, 3)  # [B, C, D, H, W]

# Step 3: Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=1, num_classes=1, level_channels=[32, 64, 128], bottleneck_channel=256).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


with torch.no_grad():
    input_tensor = img_tensor.to(device)
    output = model(input_tensor)
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    output_np = output.squeeze().cpu().numpy()

output_np = np.clip(output_np, 0, 1)  

output_sitk = sitk.GetImageFromArray(output_np.astype(np.uint8))  # or np.float32 if needed
output_sitk.CopyInformation(img_data)  # optional: preserve spacing, origin, direction
sitk.WriteImage(output_sitk, output_path + ".nii.gz")
#output_nifti = nib.Nifti1Image(output_np, affine=img_resampled.affine)
#nib.save(output_nifti, output_path)

print(f" Prediction saved to: {output_path}")
