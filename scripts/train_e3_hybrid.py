"""
WILDGUARD — E3 Controlled Training Pipeline: Hybrid Pooling & Prior Calibration
Script: scripts/train_e3_hybrid.py

Experiments supported:
  - e3_1_hybrid_pooling: HybridCNN (AvgPool + MaxPool) with capped_balanced sampler
  - e3_2_focal_loss: FocalLoss (gamma=2.0) with uniform sampling
  - e3_3_multi_offset: Multi-offset manual logging augmentation

Safeguards:
  1. Hybrid pooling tensor dimensions: (1, 64, 16, 99) -> Avg (1, 64, 1, 1) + Max (1, 64, 1, 1) -> (1, 128) -> (1, 6)
  2. Total parameters: exactly 24,294 (+384 vs SmallCNN 23,910).
  3. Capped_balanced sampler: caps Human_Sound sample weight at Gunshot_like level (eliminating the 6.9x distortion).
  4. Validation regression gates evaluated after every epoch.
  5. Canonical test set (8,189 samples) is strictly untouched during training and validation.
  6. Overwrite guard protects existing checkpoints and artifacts.
"""
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import (
    AudioDataset,
    CLASSES,
    LABEL_TO_ID,
    make_features,
    SR,
    DURATION,
    N_MELS,
    N_FFT,
    HOP,
)

MANIFEST_DEFAULT = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"

# Minority subtypes requiring balanced representation
DEFOREST_MINORITY_SUBTYPES = [
    "Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw",
    "door_wood_knock", "door_wood_creaks"
]

VEHICLE_GROUND_SUBTYPES = [
    "engine", "VehicleEngine", "car_horn", "siren", "engine_idling", "Generator"
]


class HybridCNN(nn.Module):
    """
    HybridCNN Architecture for WILDGUARD:
    Concatenates AdaptiveAvgPool2d (continuous machinery drone) and
    AdaptiveMaxPool2d (sharp 20ms transient impacts) into a 128-d feature vector.
    Exact parameter count: 24,294 (+384 parameters vs SmallCNN).
    """
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        feat = self.conv(x)
        avg_f = self.avg_pool(feat).flatten(1)
        max_f = self.max_pool(feat).flatten(1)
        pooled = torch.cat([avg_f, max_f], dim=1)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    """Focal Loss with gamma=2.0 for hard-example focusing without prior distortion."""
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_capped_balanced_weights(train_df):
    """
    Constructs capped-balanced sampling weights:
    - Base weights: 1/sqrt(class_count)
    - CAP: Human_Sound weight is capped at Gunshot_like level (prevents 6.9x prior ballooning)
    - Moderate 1.5x multiplier on real ground vehicle and manual logging subtypes.
    """
    class_counts = train_df["wildguard_class"].value_counts()
    base_weights = train_df["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    # Compute cap threshold: weight of Gunshot_like
    gunshot_base_weight = 1.0 / np.sqrt(class_counts["Gunshot_like"])
    
    # Cap Human_Sound at gunshot level
    sample_weights = base_weights.copy()
    human_mask = train_df["wildguard_class"] == "Human_Sound"
    original_human_weight = sample_weights[human_mask].iloc[0]
    sample_weights[human_mask] = gunshot_base_weight
    capped_human_weight = sample_weights[human_mask].iloc[0]

    # Apply moderate 1.5x boost to minority threat subtypes
    deforest_minority_mask = (
        (train_df["wildguard_class"] == "Deforestation") &
        (train_df["source_label"].isin(DEFOREST_MINORITY_SUBTYPES))
    )
    vehicle_ground_mask = (
        (train_df["wildguard_class"] == "Vehicle") &
        (train_df["source_label"].isin(VEHICLE_GROUND_SUBTYPES))
    )
    sample_weights[deforest_minority_mask] *= 1.5
    sample_weights[vehicle_ground_mask] *= 1.5

    print(f"  Sampler Audit:")
    print(f"    Forest_Environment weight: {sample_weights[train_df['wildguard_class']=='Forest_Environment'].iloc[0]:.6f}")
    print(f"    Deforestation base weight: {sample_weights[train_df['wildguard_class']=='Deforestation'].iloc[0]:.6f}")
    print(f"    Gunshot_like weight:       {gunshot_base_weight:.6f}")
    print(f"    Human_Sound uncapped:      {original_human_weight:.6f}")
    print(f"    Human_Sound CAPPED:        {capped_human_weight:.6f} (Capping reduction: {original_human_weight/capped_human_weight:.2f}x)")
    print(f"    Human_Sound / Forest_Env ratio: {capped_human_weight / sample_weights[train_df['wildguard_class']=='Forest_Environment'].iloc[0]:.2f}x (was 6.90x in E2)")

    return sample_weights.to_numpy(dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="WILDGUARD E3 Training Pipeline")
    parser.add_argument("--experiment", type=str, default="e3_1_hybrid_pooling",
                        choices=["e3_1_hybrid_pooling", "e3_2_focal_loss", "e3_3_multi_offset"],
                        help="Experiment identifier")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--sampler", type=str, default="capped_balanced",
                        choices=["capped_balanced", "uniform"], help="Sampling strategy")
    parser.add_argument("--loss", type=str, default="crossentropy",
                        choices=["crossentropy", "focal"], help="Loss function")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT), help="Manifest path")
    parser.add_argument("--out", type=str, default=None, help="Output directory path")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch verification without training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = ROOT / f"outputs/{args.experiment}"

    # Overwrite protection
    best_path = out_dir / "wildguard_audio_cnn_best.pt"
    latest_path = out_dir / "checkpoint_latest.pt"
    history_path = out_dir / "training_history.json"
    config_path = out_dir / "experiment_config.json"

    if best_path.exists() and not args.resume and not args.smoke_test:
        print(f"TERMINATING: Output folder '{out_dir}' already contains a checkpoint.")
        print("To protect existing artifacts, training halted. Use --resume or a different --out directory.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load manifest and verify splits
    manifest_path = Path(args.manifest)
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)

    train = df[
        (df["e2_eligible"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ].copy().reset_index(drop=True)

    val = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)

    print(f"Loaded: Train={len(train):,} samples | Validation={len(val):,} samples")
    print("\nTraining class distribution:")
    print(train["wildguard_class"].value_counts().to_string())
    print("\nValidation class distribution:")
    print(val["wildguard_class"].value_counts().to_string())

    # 2. Instantiate and verify model
    model = HybridCNN(len(CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HybridCNN (total params: {total_params:,})")
    assert total_params == 24294, f"Parameter count mismatch: {total_params}"

    # Verify forward pass shape
    sample_x, sample_y = AudioDataset(train.head(1))[0]
    with torch.no_grad():
        out = model(sample_x.unsqueeze(0).to(device))
    print(f"Input feature shape: {tuple(sample_x.shape)} | Logits output: {tuple(out.shape)}")
    assert out.shape == (1, 6), f"Output shape mismatch: {out.shape}"

    if args.smoke_test:
        print("\nSMOKE TEST COMPLETED SUCCESSFULLY.")
        return

    # 3. Setup DataLoader
    if args.sampler == "capped_balanced":
        print("\nConfiguring capped_balanced sampler...")
        weights = build_capped_balanced_weights(train)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train),
            replacement=True,
        )
        train_loader = DataLoader(
            AudioDataset(train), batch_size=args.batch_size,
            sampler=sampler, num_workers=args.workers,
        )
    else:
        print("\nConfiguring uniform DataLoader...")
        train_loader = DataLoader(
            AudioDataset(train), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers,
        )

    val_loader = DataLoader(
        AudioDataset(val), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    # 4. Setup Loss & Optimizer
    if args.loss == "focal":
        print(f"Loss: FocalLoss (gamma={args.gamma})")
        criterion = FocalLoss(gamma=args.gamma)
    else:
        print("Loss: CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Resume state if requested
    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume and latest_path.exists():
        print(f"Resuming training from {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {ckpt['epoch']}, best_val_loss={best_val_loss:.4f}")

    def evaluate(loader):
        model.eval()
        loss_sum, total = 0.0, 0
        all_y, all_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * len(y)
                total += len(y)
                all_y.extend(y.cpu().numpy().tolist())
                all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        avg_loss = loss_sum / max(1, total)
        report = classification_report(all_y, all_pred, target_names=CLASSES, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_y, all_pred, labels=list(range(len(CLASSES))))
        return avg_loss, report, cm

    print("\n" + "=" * 70)
    print(f"Starting Training: {args.experiment} (Epochs {start_epoch} to {args.epochs})")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

            if batch_idx % 250 == 0 or batch_idx == len(train_loader):
                cur_train_loss = loss_sum / max(1, total)
                cur_train_acc = correct / max(1, total)
                print(f"  [Epoch {epoch} | Step {batch_idx}/{len(train_loader)}] loss={cur_train_loss:.4f} acc={cur_train_acc:.4f}", flush=True)

        val_loss, val_rep, val_cm = evaluate(val_loader)
        scheduler.step(val_loss)

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = val_rep["accuracy"]
        val_macro_f1 = val_rep["macro avg"]["f1-score"]
        val_weighted_f1 = val_rep["weighted avg"]["f1-score"]
        lr = optimizer.param_groups[0]["lr"]

        # Regression gate checks
        val_gunshot_rec = val_rep["Gunshot_like"]["recall"]
        val_deforest_rec = val_rep["Deforestation"]["recall"]
        val_human_prec = val_rep["Human_Sound"]["precision"]
        val_vehicle_prec = val_rep["Vehicle"]["precision"]

        print(
            f"\n>>> Epoch {epoch}/{args.epochs} Summary: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} macro_f1={val_macro_f1:.4f} | lr={lr:.6g}",
            flush=True,
        )
        print(f"    Regression Gates Check:")
        print(f"      Gunshot Recall:      {val_gunshot_rec*100:.2f}% (Gate >= 94.0%: {'PASS' if val_gunshot_rec>=0.94 else 'FAIL'})")
        print(f"      Deforest Recall:     {val_deforest_rec*100:.2f}% (Gate >= 87.5%: {'PASS' if val_deforest_rec>=0.875 else 'FAIL'})")
        print(f"      Human Precision:     {val_human_prec*100:.2f}% (Gate >= 40.0%: {'PASS' if val_human_prec>=0.40 else 'FAIL'})")
        print(f"      Vehicle Precision:   {val_vehicle_prec*100:.2f}% (Gate >= 55.0%: {'PASS' if val_vehicle_prec>=0.55 else 'FAIL'})")

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_macro_f1": round(val_macro_f1, 5),
            "val_weighted_f1": round(val_weighted_f1, 5),
            "val_gunshot_recall": round(val_gunshot_rec, 5),
            "val_deforest_recall": round(val_deforest_rec, 5),
            "val_human_precision": round(val_human_prec, 5),
            "val_vehicle_precision": round(val_vehicle_prec, 5),
            "lr": lr,
            "per_class_val": {
                c: {
                    "precision": round(val_rep[c]["precision"], 4),
                    "recall": round(val_rep[c]["recall"], 4),
                    "f1": round(val_rep[c]["f1-score"], 4)
                } for c in CLASSES
            }
        }
        history.append(epoch_stats)

        # Save checkpoint_latest.pt
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": CLASSES,
            "sample_rate": SR,
            "duration_seconds": DURATION,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "architecture": "HybridCNN",
            "parameters": total_params,
            "experiment": args.experiment,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        }, latest_path)

        # Save wildguard_audio_cnn_best.pt strictly on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "classes": CLASSES,
                "sample_rate": SR,
                "duration_seconds": DURATION,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP,
                "architecture": "HybridCNN",
                "parameters": total_params,
                "experiment": args.experiment,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "history": history,
            }, best_path)
            print(f"  *** NEW BEST VALIDATION CHECKPOINT SAVED: Epoch {epoch} (val_loss={val_loss:.4f}, macro_f1={val_macro_f1:.4f}) ***\n")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save full experiment config
    exp_config = {
        "experiment": args.experiment,
        "manifest": str(manifest_path),
        "architecture": "HybridCNN",
        "parameters": total_params,
        "hardware_claim_disclaimer": "ESP32-S3 memory is a theoretical estimate (398 KB INT8 peak activation for Layer 0 Conv1); actual target-runtime peak memory depends on ESP-DL / TFLite Micro runtime buffer allocation and must be measured on hardware.",
        "classes": CLASSES,
        "class_index": LABEL_TO_ID,
        "preprocessing": {
            "sample_rate": SR,
            "duration_seconds": DURATION,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP,
            "fmin": 40,
            "fmax": 8000
        },
        "sampler": args.sampler,
        "loss": args.loss,
        "epochs": args.epochs,
        "best_epoch": next((h["epoch"] for h in history if h["val_loss"] == best_val_loss), None),
        "best_val_loss": best_val_loss,
        "history": history
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED.")
    print(f"Best validation checkpoint: {best_path} (val_loss={best_val_loss:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
