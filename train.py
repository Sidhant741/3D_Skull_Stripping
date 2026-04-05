import math
import os
import torch
from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,
    BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from torch.nn import CrossEntropyLoss
from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (
    train_transform, train_transform_cuda,
    val_transform, val_transform_cuda
)
from tqdm import tqdm

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.3):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        smooth = 1e-6
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# Dice Score function
def dice_score(output, label):
    smooth = 1e-6  # small constant to avoid division by zero
    output = output.sigmoid()  # apply sigmoid to get probabilities
    output = (output > 0.5).float()  # threshold to get binary values
    intersection = (output * label).sum()
    return (2. * intersection + smooth) / (output.sum() + label.sum() + smooth)
    
def soft_dice_score(output, label):
    smooth = 1e-6
    output = torch.sigmoid(output)
    intersection = (output * label).sum()
    return (2. * intersection + smooth) / (output.sum() + label.sum() + smooth)

if BACKGROUND_AS_CLASS:
    NUM_CLASSES += 1

# TensorBoard writer
writer = SummaryWriter("runs/unet3d_32_64_128")

# Model init
# model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
model = UNet3D(in_channels=1, num_classes=1, level_channels=[32, 64, 128], bottleneck_channel=256)

# Device
device = torch.device("cuda" if TRAIN_CUDA and torch.cuda.is_available() else "cpu")
if TRAIN_CUDA and not torch.cuda.is_available():
    print("CUDA not available! Training on CPU...")

# Set transforms based on CUDA
train_transforms = train_transform
val_transforms = val_transform
if device.type == "cuda":
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda

model = model.to(device)

# Dataloaders
train_loader, val_loader, _ = get_train_val_test_Dataloaders(
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms
)

# Loss function & optimizer
# criterion = CrossEntropyLoss(weight=torch.tensor(BCE_WEIGHTS))
pos_weight = torch.tensor([BCE_WEIGHTS[1]/BCE_WEIGHTS[0]], device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = Adam(params=model.parameters())


# Training loop
min_val_loss = math.inf
os.makedirs("checkpoints", exist_ok=True)



for epoch in range(TRAINING_EPOCH):
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_soft_dice = 0.0


    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
        image, label = batch["image"], batch["label"]
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(image)
        # print(output.shape)
        # print(output.dtype, image.dtype, label.dtype)
        # print(output.device, label.device)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice_score(output, label).item()
        train_soft_dice += soft_dice_score(output, label).item()

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_soft_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", unit="batch"):
            image, label = batch["image"], batch["label"]
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            val_loss += loss.item()
            val_dice += dice_score(output, label).item()
            val_soft_dice += soft_dice_score(output, label).item()


    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    avg_train_soft_dice = train_soft_dice / len(train_loader)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_soft_dice = val_soft_dice / len(val_loader)


    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Dice/Train", avg_train_dice, epoch)
    writer.add_scalar("Dice/Validation", avg_val_dice, epoch)
    writer.add_scalar("Soft Dice/Train", avg_train_soft_dice, epoch)
    writer.add_scalar("Soft Dice/Validation", avg_val_soft_dice, epoch)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    print(f"Epoch {epoch+1}: Train Dice = {avg_train_dice:.6f}, Val Dice = {avg_val_dice:.6f}")
    print(f"Epoch {epoch+1}: Train Soft Dice = {avg_train_soft_dice:.6f}, Val Soft Dice = {avg_val_soft_dice:.6f}")


    if avg_val_loss < min_val_loss:
        print(f"Validation loss improved ({min_val_loss:.6f} → {avg_val_loss:.6f}) — saving model.")
        min_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"checkpoints/epoch{epoch:03d}_valLoss{avg_val_loss:.6f}.pth")
    else:
        print(f"No improvement in val loss ({avg_val_loss:.6f})")

writer.flush()
writer.close()
