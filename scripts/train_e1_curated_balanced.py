"""
WILDGUARD — E1 Controlled Training Experiment: audio_curated_subclass_balanced_v1

Changes vs baseline (audio_hard_negative_experiment_rerun):
  1. Manifest: master_manifest_v14_e1_curated.csv
     - Excludes airplane, helicopter, Helicopter, train from Vehicle (training only)
     - Excludes sea_waves from Forest_Environment (training only)
     - Val/test splits are identical to baseline for a fair comparison
  2. Sampler: WeightedRandomSampler with 1/sqrt(class_count) base weights
     + 2x multiplier for non-chainsaw Deforestation and ground-engine Vehicle subtypes
  3. Architecture, preprocessing, class mapping, random seed: UNCHANGED from baseline.

Preprocessing contract (must match baseline checkpoint):
  SR=16000, DURATION=4s, N_MELS=64, N_FFT=400, HOP=160, fmin=40, fmax=8000
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# Import shared architecture from baseline training script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import AudioDataset, SmallCNN, CLASSES, LABEL_TO_ID, make_features, SR, DURATION, N_MELS, N_FFT, HOP

MANIFEST_E1 = ROOT / "data/processed/master_manifest_v14_e1_curated.csv"
OUT_DEFAULT = ROOT / "outputs/audio_curated_subclass_balanced_v1"

# Non-chainsaw Deforestation subtypes that need boosted sampling
DEFOREST_MINORITY_SUBTYPES = ["Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw"]
# Ground vehicle subtypes (non-aerial, non-rail)
VEHICLE_GROUND_SUBTYPES = ["engine", "VehicleEngine", "car_horn", "siren"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Verify data loading without training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load E1 manifest ---
    df = pd.read_csv(MANIFEST_E1, dtype=str, low_memory=False)

    # Training split: eligible AND e1_eligible
    train = df[
        (df["e1_eligible"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ].copy().reset_index(drop=True)

    # Val/test: any eligible row regardless of e1_eligible (includes aircraft for unchanged comparison)
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

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Train class distribution:")
    print(train["wildguard_class"].value_counts().to_string())

    # Verify all audio files exist
    all_paths = list(train["audio_path"]) + list(val["audio_path"]) + list(test["audio_path"])
    missing_files = [p for p in all_paths if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files missing. First: {missing_files[0]}"
        )
    print(f"All {len(all_paths)} audio files verified.")

    # Smoke-test: verify feature shape and model output
    model = SmallCNN(len(CLASSES)).to(device)
    sample_x, sample_y = AudioDataset(train.head(1))[0]
    with torch.no_grad():
        out = model(sample_x.unsqueeze(0).to(device))
    print(f"Feature shape: {tuple(sample_x.shape)}, Model output shape: {tuple(out.shape)}")

    if args.smoke_test:
        print("SMOKE TEST PASSED. Full training has not started.")
        return

    out_dir = Path(args.out)
    best_path = out_dir / "wildguard_audio_cnn_best.pt"
    if best_path.exists():
        print(f"TERMINATING: Checkpoint already exists at {best_path}")
        print("Delete output folder or use a different --out path to rerun.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Weighted Sampler ---
    # Base: 1/sqrt(class_count) for all training samples
    class_counts = train["wildguard_class"].value_counts()
    base_weights = train["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    # Boost: 2x for non-chainsaw Deforestation (0% accuracy at baseline)
    # Reduced from 4x (attempt 1) which caused val_loss instability at epoch 5
    deforest_minority_mask = (
        (train["wildguard_class"] == "Deforestation") &
        (train["source_label"].isin(DEFOREST_MINORITY_SUBTYPES))
    )
    # Boost: 2x for ground Vehicle subtypes (44-83% accuracy, only 154 train clips)
    # Reduced from 4x (attempt 1) which collapsed Vehicle precision to 21%
    vehicle_ground_mask = (
        (train["wildguard_class"] == "Vehicle") &
        (train["source_label"].isin(VEHICLE_GROUND_SUBTYPES))
    )

    sample_weights = base_weights.copy()
    sample_weights[deforest_minority_mask] *= 2.0
    sample_weights[vehicle_ground_mask] *= 2.0
    sample_weights_np = sample_weights.to_numpy(dtype=np.float64)

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
    test_loader = DataLoader(
        AudioDataset(test), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

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

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        val_loss, val_y, val_pred = evaluate(val_loader)
        scheduler.step(val_loss)

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = sum(a == b for a, b in zip(val_y, val_pred)) / max(1, len(val_y))
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr:.6g}",
            flush=True,
        )
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5), "lr": lr,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "sample_rate": SR,
                "duration_seconds": DURATION,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP,
                "experiment": "audio_curated_subclass_balanced_v1",
                "epoch": epoch,
            }, best_path)
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save training history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- Evaluate on test set ---
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, y_true, y_pred = evaluate(test_loader)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True, zero_division=0,
    )

    print(f"\nBest val_loss: {round(best_val, 4)}")
    print(f"Held-out test loss: {round(test_loss, 4)}")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv")

    with open(out_dir / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Best model:", best_path)
    print("Test report:", out_dir / "test_report.json")
    print("Confusion matrix:", out_dir / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
