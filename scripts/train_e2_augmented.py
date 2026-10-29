"""
WILDGUARD — E2 Controlled Training Experiment: audio_curated_augmented_e2

Enhancements vs E1 / Baseline:
  1. Manifest: master_manifest_v15_e2_augmented.csv
     - Includes 290 targeted UrbanSound8K 'engine_idling' recordings (240 train, 50 val, 0 test)
     - Includes 68 FSC22 combustion 'Generator' recordings (63 train, 5 val, 0 test)
     - Includes 80 ESC-50 manual logging recordings: 'door_wood_knock' & 'door_wood_creaks' (64 train, 16 val, 0 test)
     - Includes 40 ESC-50 water runoff geophony: 'pouring_water' (32 train, 8 val, 0 test)
     - Canonical test set (8,189 samples) is 100% frozen, identical to v13/v14.
  2. Sampler: WeightedRandomSampler with 1/sqrt(class_count) base weights
     + 1.5x multiplier for non-chainsaw Deforestation and ground-engine Vehicle subtypes.
  3. Preprocessing, architecture, loss, optimizer, and random seed:
     SR=16000, DURATION=4s, N_MELS=64, N_FFT=400, HOP=160, fmin=40, fmax=8000
     SmallCNN (identical to baseline).
  4. Resumable Checkpointing:
     Saves model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_val_loss, history.
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

# Import shared architecture and preprocessing from baseline script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import (
    AudioDataset,
    SmallCNN,
    CLASSES,
    LABEL_TO_ID,
    make_features,
    SR,
    DURATION,
    N_MELS,
    N_FFT,
    HOP,
)

MANIFEST_E2 = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
OUT_DEFAULT = ROOT / "outputs/audio_curated_augmented_e2"

# Non-chainsaw Deforestation subtypes that need balanced representation
DEFOREST_MINORITY_SUBTYPES = [
    "Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw",
    "door_wood_knock", "door_wood_creaks"
]

# Ground vehicle subtypes (non-aerial, non-rail combustion noise)
VEHICLE_GROUND_SUBTYPES = [
    "engine", "VehicleEngine", "car_horn", "siren", "engine_idling", "Generator"
]


def parse_args():
    parser = argparse.ArgumentParser(description="WILDGUARD E2 Training")
    parser.add_argument("--smoke-test", action="store_true", help="Verify data loading without training")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--multiplier", type=float, default=1.5, help="Subtype sampling multiplier")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT), help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint_latest.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "wildguard_audio_cnn_best.pt"
    latest_path = out_dir / "checkpoint_latest.pt"
    history_path = out_dir / "training_history.json"
    config_path = out_dir / "experiment_config.json"

    # --- Load E2 manifest ---
    print(f"Loading E2 manifest: {MANIFEST_E2}")
    df = pd.read_csv(MANIFEST_E2, dtype=str, low_memory=False)

    # Filter by e2_eligible flag
    train = df[
        (df["e2_eligible"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ].copy().reset_index(drop=True)

    val = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)

    test = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "test")
    ].copy().reset_index(drop=True)

    # Validate class presence
    target_classes = set(CLASSES)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        present = set(split_df["wildguard_class"].unique())
        missing = target_classes - present
        if missing:
            raise RuntimeError(f"Split '{split_name}' missing classes: {missing}")

    print(f"Train samples: {len(train)} | Val samples: {len(val)} | Test samples: {len(test)}")
    assert len(test) == 8189, f"Canonical test set mismatch! Expected 8189, got {len(test)}"

    print("\nTrain class distribution:")
    print(train["wildguard_class"].value_counts().to_string())
    print("\nValidation class distribution:")
    print(val["wildguard_class"].value_counts().to_string())

    # Smoke-test verify
    model = SmallCNN(len(CLASSES)).to(device)
    sample_x, sample_y = AudioDataset(train.head(1))[0]
    with torch.no_grad():
        out = model(sample_x.unsqueeze(0).to(device))
    print(f"\nFeature verification: shape={tuple(sample_x.shape)}, model_out={tuple(out.shape)}")

    if args.smoke_test:
        print("SMOKE TEST COMPLETED SUCCESSFULLY.")
        return

    # --- Weighted Sampler ---
    class_counts = train["wildguard_class"].value_counts()
    base_weights = train["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    deforest_minority_mask = (
        (train["wildguard_class"] == "Deforestation") &
        (train["source_label"].isin(DEFOREST_MINORITY_SUBTYPES))
    )
    vehicle_ground_mask = (
        (train["wildguard_class"] == "Vehicle") &
        (train["source_label"].isin(VEHICLE_GROUND_SUBTYPES))
    )

    sample_weights = base_weights.copy()
    sample_weights[deforest_minority_mask] *= args.multiplier
    sample_weights[vehicle_ground_mask] *= args.multiplier
    sample_weights_np = sample_weights.to_numpy(dtype=np.float64)

    print(f"\nSampler configuration:")
    print(f"  Base weights: 1/sqrt(class_count)")
    print(f"  Deforestation minority boost: {args.multiplier}x ({deforest_minority_mask.sum()} samples)")
    print(f"  Ground Vehicle boost: {args.multiplier}x ({vehicle_ground_mask.sum()} samples)")

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights_np, dtype=torch.double),
        num_samples=len(train),
        replacement=True,
    )

    train_loader = DataLoader(
        AudioDataset(train), batch_size=args.batch_size,
        sampler=sampler, num_workers=args.workers,
    )
    val_loader = DataLoader(
        AudioDataset(val), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    start_epoch = 1
    best_val = float("inf")
    history = []

    # Resume from checkpoint if requested
    if args.resume and latest_path.exists():
        print(f"Resuming training from {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {ckpt['epoch']}, best_val_loss={best_val:.4f}")

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
        return loss_sum / max(1, total), all_y, all_pred

    print("\n" + "=" * 70)
    print(f"Starting E2 Training: Epochs {start_epoch} to {args.epochs}")
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

        val_loss, val_y, val_pred = evaluate(val_loader)
        scheduler.step(val_loss)

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = sum(a == b for a, b in zip(val_y, val_pred)) / max(1, len(val_y))
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n>>> Epoch {epoch}/{args.epochs} Summary: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.6g}",
            flush=True,
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": lr,
        }
        history.append(epoch_stats)

        # Always save latest checkpoint (resumable)
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
            "experiment": "audio_curated_augmented_e2",
            "epoch": epoch,
            "best_val_loss": best_val,
            "history": history,
        }, latest_path)

        # Save best checkpoint strictly based on lowest validation loss
        if val_loss < best_val:
            best_val = val_loss
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
                "experiment": "audio_curated_augmented_e2",
                "epoch": epoch,
                "best_val_loss": best_val,
                "history": history,
            }, best_path)
            print(f"  *** NEW BEST CHECKPOINT SAVED: Epoch {epoch} (val_loss={val_loss:.4f}, val_acc={val_acc:.4f}) ***\n")

        # Save history json
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save experiment config
    exp_config = {
        "experiment": "audio_curated_augmented_e2",
        "manifest": str(MANIFEST_E2),
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
        "multiplier": args.multiplier,
        "epochs": args.epochs,
        "best_epoch": next((h["epoch"] for h in history if h["val_loss"] == best_val), None),
        "best_val_loss": best_val,
        "history": history
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED.")
    print(f"Best checkpoint: {best_path} (val_loss={best_val:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
