"""
Distributed Data Parallel (DDP) training for UNet3DDeep skull stripping.

Launch with torchrun (recommended):
    torchrun --nproc_per_node=NUM_GPUS train.py

Single-node multi-GPU example (4 GPUs):
    torchrun --nproc_per_node=4 train.py

Single GPU (still works — DDP with 1 process is identical to non-DDP):
    torchrun --nproc_per_node=1 train.py
    # or just:
    python train.py   ← falls back to single-GPU non-DDP automatically
"""

import math
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,
    BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from dataset import SkullStrippingDataset
from unet3d import UNet3DDeep
from transforms import (
    train_transform, train_transform_cuda,
    val_transform, val_transform_cuda,
)
from config import (
    DATASET_PATH, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def is_ddp() -> bool:
    """True when launched via torchrun / torch.distributed."""
    return dist.is_available() and dist.is_initialized()


def global_rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1


def is_main() -> bool:
    """Only rank-0 should log, save checkpoints, write TensorBoard."""
    return global_rank() == 0


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor and divide by world size."""
    if not is_ddp():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size()


# ------------------------------------------------------------------ #
# Loss                                                                #
# ------------------------------------------------------------------ #

class BCEDiceLoss(torch.nn.Module):
    """10% BCE + 90% soft Dice — Dice dominates for class-imbalanced segmentation."""
    def __init__(self, bce_weight: float = 0.1):
        super().__init__()
        self.bce        = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        smooth   = 1e-6
        probs    = torch.sigmoid(inputs)
        dims     = tuple(range(1, probs.ndim))
        inter    = (probs * targets).sum(dim=dims)
        denom    = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #

@torch.no_grad()
def dice_score(output, label, eps: float = 1e-6) -> float:
    with torch.cuda.amp.autocast(enabled=False):
        probs = torch.sigmoid(output).float()
        preds = (probs > 0.5).float()
        label = label.float()
        dims  = tuple(range(1, preds.ndim))
        inter = (preds * label).sum(dim=dims)
        denom = preds.sum(dim=dims) + label.sum(dim=dims)
        return ((2.0 * inter + eps) / (denom + eps)).mean().item()


@torch.no_grad()
def soft_dice_score(output, label, eps: float = 1e-6) -> float:
    with torch.cuda.amp.autocast(enabled=False):
        probs = torch.sigmoid(output).float()
        label = label.float()
        dims  = tuple(range(1, probs.ndim))
        inter = (probs * label).sum(dim=dims)
        denom = probs.sum(dim=dims) + label.sum(dim=dims)
        return ((2.0 * inter + eps) / (denom + eps)).mean().item()


# ------------------------------------------------------------------ #
# Main training function (runs on every rank)                         #
# ------------------------------------------------------------------ #

def main():
    # ---- DDP init ------------------------------------------------- #
    # torchrun sets LOCAL_RANK automatically; fall back to 0 for plain
    # `python train.py` single-GPU runs.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp    = "LOCAL_RANK" in os.environ and torch.cuda.device_count() > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main():
        print(f"World size : {world_size()}")
        print(f"Device     : {device}")

    # ---- Transforms ----------------------------------------------- #
    # RandAffined / RandGaussianNoised run on CPU (MONAI limitation),
    # so we always use the CPU transform pipeline for training and move
    # tensors to device in the loop with non_blocking=True.
    # Val transforms use the cuda variant only when NOT using DDP
    # (DDP + MONAI cuda tensors + DistributedSampler causes deadlocks).
    _train_tf = train_transform          # CPU augmentation pipeline
    _val_tf   = val_transform            # CPU val pipeline

    # ---- Datasets & samplers -------------------------------------- #
    train_ds = SkullStrippingDataset(DATASET_PATH, mode='train', transform=_train_tf)
    val_ds   = SkullStrippingDataset(DATASET_PATH, mode='valid', transform=_val_tf)

    # DistributedSampler shards the dataset across ranks.
    # shuffle=True is handled by sampler.set_epoch(epoch) each epoch.
    train_sampler = DistributedSampler(train_ds, shuffle=True)  if use_ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),   # only shuffle when no sampler
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # ---- Model ---------------------------------------------------- #
    if BACKGROUND_AS_CLASS:
        num_classes = NUM_CLASSES + 1
    else:
        num_classes = NUM_CLASSES

    model = UNet3DDeep(
        in_channels=1,
        num_classes=num_classes,
        level_channels=[32, 64, 128, 256],
        bottleneck_channel=512,
    ).to(device)

    if use_ddp:
        # find_unused_parameters=False is faster; set True only if you
        # have branches that aren't always executed (we don't).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Optimizer / scheduler / scaler --------------------------- #
    criterion = BCEDiceLoss(bce_weight=0.1)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # ReduceLROnPlateau needs the globally-averaged val loss, which we
    # all-reduce below — so it's safe to call on every rank.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=is_main())
    scaler    = GradScaler(enabled=device.type == "cuda")

    # ---- TensorBoard (rank 0 only) -------------------------------- #
    writer = SummaryWriter("runs/unet3d_deep_ddp") if is_main() else None

    # ---- Checkpoint state ----------------------------------------- #
    min_val_loss   = math.inf
    best_ckpt_path = None
    os.makedirs("checkpoints", exist_ok=True)

    # ================================================================ #
    # Epoch loop                                                        #
    # ================================================================ #
    for epoch in range(TRAINING_EPOCH):

        # Tell sampler the epoch so each rank gets a different shuffle
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ------------------------------------------------ #
        model.train()
        train_loss_sum, train_dice_sum, train_soft_dice_sum = 0.0, 0.0, 0.0
        num_train_batches = 0

        train_iter = tqdm(
            train_loader,
            desc=f"[rank {global_rank()}] Train {epoch+1}",
            unit="batch",
            disable=not is_main(),   # only rank-0 shows progress bar
        )

        for batch in train_iter:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True).float().clamp_(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                output, aux = model(image)                        # train → (main, aux)
                label_down  = F.interpolate(label, size=aux.shape[2:], mode='nearest')
                loss_main   = criterion(output, label)
                loss_aux    = criterion(aux, label_down)
                loss        = loss_main + 0.4 * loss_aux

            if not torch.isfinite(loss):
                if is_main():
                    print(f"[Epoch {epoch+1}] Skipping batch — non-finite loss")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum      += loss_main.item()
            train_dice_sum      += dice_score(output.detach(), label)
            train_soft_dice_sum += soft_dice_score(output.detach(), label)
            num_train_batches   += 1

        # ---- Validate --------------------------------------------- #
        model.eval()
        val_loss_sum, val_dice_sum, val_soft_dice_sum = 0.0, 0.0, 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"[rank {global_rank()}] Val   {epoch+1}",
                unit="batch",
                disable=not is_main(),
            ):
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True).float().clamp_(0.0, 1.0)

                with autocast(enabled=device.type == "cuda"):
                    output = model(image)                         # eval → main only
                    loss   = criterion(output, label)

                if not torch.isfinite(loss):
                    continue

                val_loss_sum      += loss.item()
                val_dice_sum      += dice_score(output, label)
                val_soft_dice_sum += soft_dice_score(output, label)
                num_val_batches   += 1

        # ---- All-reduce metrics across ranks ---------------------- #
        # Convert local sums to tensors, all-reduce (sum), then average.
        # This gives the true global average across all GPUs.
        def _avg(total, count):
            t = torch.tensor([total, float(count)], device=device)
            reduce_mean(t)           # in-place all-reduce + /world_size
            # After reduce_mean: t[0] = global_sum/world_size,
            #                    t[1] = global_count/world_size
            # Dividing gives the correct global average.
            return (t[0] / t[1].clamp(min=1)).item()

        avg_train_loss      = _avg(train_loss_sum,      num_train_batches)
        avg_train_dice      = _avg(train_dice_sum,      num_train_batches)
        avg_train_soft_dice = _avg(train_soft_dice_sum, num_train_batches)
        avg_val_loss        = _avg(val_loss_sum,        num_val_batches)
        avg_val_dice        = _avg(val_dice_sum,        num_val_batches)
        avg_val_soft_dice   = _avg(val_soft_dice_sum,   num_val_batches)

        # Scheduler uses globally-averaged val loss — safe on all ranks
        scheduler.step(avg_val_loss)

        # ---- Logging & checkpointing (rank 0 only) ---------------- #
        if is_main():
            writer.add_scalar("Loss/Train",           avg_train_loss,      epoch)
            writer.add_scalar("Loss/Validation",      avg_val_loss,        epoch)
            writer.add_scalar("Dice/Train",           avg_train_dice,      epoch)
            writer.add_scalar("Dice/Validation",      avg_val_dice,        epoch)
            writer.add_scalar("Soft Dice/Train",      avg_train_soft_dice, epoch)
            writer.add_scalar("Soft Dice/Validation", avg_val_soft_dice,   epoch)
            writer.add_scalar("LR",                   optimizer.param_groups[0]['lr'], epoch)

            print(f"\nEpoch {epoch+1}/{TRAINING_EPOCH}")
            print(f"  Train — Loss: {avg_train_loss:.6f} | Dice: {avg_train_dice:.6f} | Soft Dice: {avg_train_soft_dice:.6f}")
            print(f"  Val   — Loss: {avg_val_loss:.6f} | Dice: {avg_val_dice:.6f} | Soft Dice: {avg_val_soft_dice:.6f}")
            print(f"  LR    — {optimizer.param_groups[0]['lr']:.2e}")

            if avg_val_loss < min_val_loss:
                print(f"  Val loss improved ({min_val_loss:.6f} → {avg_val_loss:.6f}) — saving.")
                min_val_loss = avg_val_loss
                # Save the underlying module weights (not the DDP wrapper)
                raw_model  = model.module if use_ddp else model
                new_ckpt   = f"checkpoints/best_epoch{epoch:03d}_valLoss{avg_val_loss:.6f}.pth"
                torch.save(raw_model.state_dict(), new_ckpt)
                if best_ckpt_path and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = new_ckpt
            else:
                print(f"  No improvement ({avg_val_loss:.6f} ≥ {min_val_loss:.6f})")

    # ---- Cleanup -------------------------------------------------- #
    if is_main():
        writer.flush()
        writer.close()
        print(f"\nTraining complete. Best val loss : {min_val_loss:.6f}")
        print(f"Best checkpoint            : {best_ckpt_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()