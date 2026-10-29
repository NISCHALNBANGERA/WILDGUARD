"""
WILDGUARD Experiment E3.5 Training Pipeline:
Calibrated Focal Loss (gamma=1.0) + Principled Minority Sampling (Uncapped Human_Sound)
Architecture: HybridCNN (24,294 parameters)
Manifest: data/processed/master_manifest_v16_e3_cleaned.csv (8,210 leak-free validation samples)

Data-Driven Rationales for E3.5:
1. Human_Sound Uncapped: In E3.4, capping Human_Sound at the Gunshot level (0.020947 vs natural 0.047298)
   cut Human_Sound gradient representation by 55.7%, causing Human recall to crash from 84.4% to 62.5% on validation.
   E3.5 restores Human_Sound to its mathematically principled 1/sqrt(N_human) weight (0.047298).
2. Vehicle Prior Capped: Preserves the Vehicle weight cap at Gunshot level (0.020947) and the 2.5x ambient
   weather rumble boost (thunderstorm, wind, rain, etc.) which successfully cut Vehicle false alarms by 71% in E3.4.
3. Linear Focal Loss (gamma=1.0): In E3.4, quadratic focal attenuation (gamma=2.0) downweighted easy ambient
   rainforest biophony by up to 70x, blinding the model to ambient false alarms and causing 522 RFCx environment
   clips to be misclassified as Deforestation. Linear Focal Loss (gamma=1.0) retains 10x greater gradient pressure
   on ambient background biophony while maintaining focal gradient assistance on low-SNR threats.
4. Exact Same 8,210 Validation Samples: Zero contamination, frozen canonical test set (8,189 samples).
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from train_e3_hybrid import HybridCNN, FocalLoss

MANIFEST_DEFAULT = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"

DEFOREST_MINORITY_SUBTYPES = [
    "Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw",
    "door_wood_knock", "door_wood_creaks"
]

RUMBLE_HARD_NEGATIVES = [
    "thunderstorm", "wind", "sea_waves", "rain", "silence", "water_drops", "water"
]


def build_e3_5_calibrated_weights(train_df):
    """
    Constructs calibrated sampling weights:
    - Base weights: 1/sqrt(class_count)
    - Gunshot weight is the universal cap for Vehicle only (prevents vehicle prior explosion)
    - Human_Sound remains UNCAPPED at natural 1/sqrt(N_human) (0.047298) to prevent recall starvation
    - Boost ambient weather rumble hard-negatives in Forest_Environment by 2.5x
    - Boost manual logging minority threat subtypes by 1.5x
    """
    class_counts = train_df["wildguard_class"].value_counts()
    base_weights = train_df["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    gunshot_cap = 1.0 / np.sqrt(class_counts["Gunshot_like"])
    sample_weights = base_weights.copy()

    # 1. Vehicle Capped at Gunshot level (preserves E3.4 vehicle precision fix)
    vehicle_mask = train_df["wildguard_class"] == "Vehicle"
    sample_weights[vehicle_mask] = gunshot_cap

    # 2. Human_Sound is INTENTIONALLY UNCAPPED: remains at natural 1/sqrt(N_human) = 0.047298
    # This prevents the 55.7% representation deficit that collapsed Human recall in E3.4

    # 3. Boost ambient weather rumble in Forest_Environment by 2.5x
    rumble_mask = (
        (train_df["wildguard_class"] == "Forest_Environment") &
        (train_df["source_label"].astype(str).str.lower().isin(RUMBLE_HARD_NEGATIVES))
    )
    sample_weights[rumble_mask] *= 2.5

    # 4. Boost manual logging subtypes in Deforestation by 1.5x
    logging_mask = (
        (train_df["wildguard_class"] == "Deforestation") &
        (train_df["source_label"].isin(DEFOREST_MINORITY_SUBTYPES))
    )
    sample_weights[logging_mask] *= 1.5

    non_rumble_mask = (train_df["wildguard_class"] == "Forest_Environment") & (~rumble_mask)
    forest_base_weight = sample_weights[non_rumble_mask].iloc[0]
    forest_rumble_weight = sample_weights[rumble_mask].iloc[0]
    human_weight = sample_weights[train_df["wildguard_class"] == "Human_Sound"].iloc[0]
    vehicle_weight = sample_weights[vehicle_mask].iloc[0]
    gunshot_weight = sample_weights[train_df["wildguard_class"] == "Gunshot_like"].iloc[0]

    print(f"\n  === E3.5 Calibrated Sampler Audit ===")
    print(f"    Forest_Environment base weight:   {forest_base_weight:.6f}")
    print(f"    Forest_Environment rumble weight: {forest_rumble_weight:.6f} ({forest_rumble_weight/forest_base_weight:.1f}x)")
    print(f"    Gunshot_like weight:              {gunshot_weight:.6f}")
    print(f"    Human_Sound weight (UNCAPPED):    {human_weight:.6f} ({human_weight/gunshot_weight:.2f}x over Gunshot)")
    print(f"    Vehicle weight (CAPPED):          {vehicle_weight:.6f} ({vehicle_weight/gunshot_weight:.2f}x over Gunshot)")
    print(f"  ====================================\n")

    return sample_weights.to_numpy(dtype=np.float64)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_y.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    report = classification_report(
        all_y, all_preds, target_names=CLASSES, output_dict=True, digits=5, zero_division=0
    )
    cm = confusion_matrix(all_y, all_preds, labels=range(len(CLASSES)))
    macro_f1 = float(report["macro avg"]["f1-score"])

    # Regression Gates Verification
    gunshot_rec = float(report["Gunshot_like"]["recall"])
    deforest_rec = float(report["Deforestation"]["recall"])
    human_prec = float(report["Human_Sound"]["precision"])
    vehicle_prec = float(report["Vehicle"]["precision"])

    gate_gunshot = gunshot_rec >= 0.940
    gate_deforest = deforest_rec >= 0.875
    gate_human = human_prec >= 0.400
    gate_vehicle = vehicle_prec >= 0.550
    all_gates = gate_gunshot and gate_deforest and gate_human and gate_vehicle

    metrics = {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "accuracy": float(report["accuracy"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "gunshot_recall": gunshot_rec,
        "deforest_recall": deforest_rec,
        "human_precision": human_prec,
        "vehicle_precision": vehicle_prec,
        "all_gates_pass": all_gates,
        "report": report,
        "cm": cm.tolist(),
    }
    return metrics, cm, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs/e3_5_calibrated_hybrid"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0, help="Linear focal loss exponent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("  WILDGUARD Experiment E3.5: Calibrated HybridCNN Pipeline")
    print("=" * 80)
    print(f"  Manifest:    {args.manifest}")
    print(f"  Output Dir:  {out_dir}")
    print(f"  Architecture: HybridCNN (24,294 parameters)")
    print(f"  Loss:        FocalLoss (gamma={args.gamma})")
    print(f"  Sampler:     Calibrated Hard-Negative (Human UNCAPPED, Vehicle CAPPED)")
    print(f"  Epochs:      {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")

    # Load Manifest
    df = pd.read_csv(args.manifest, dtype=str, low_memory=False)
    train_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ].copy().reset_index(drop=True)

    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)

    print(f"\n  Loaded Train:      {len(train_df):,} samples")
    print(f"  Loaded Validation: {len(val_df):,} samples")
    assert len(val_df) == 8210, f"Expected 8210 validation samples, got {len(val_df)}"

    if args.smoke_test:
        train_df = train_df.head(128).copy()
        val_df = val_df.head(64).copy()
        args.epochs = 2
        print("  SMOKE TEST MODE ENABLED.")

    # Build Calibrated Sampler
    sample_weights = build_e3_5_calibrated_weights(train_df)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True
    )

    train_loader = DataLoader(
        AudioDataset(train_df),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        AudioDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute Device: {device}")

    model = HybridCNN(n_classes=len(CLASSES)).to(device)
    criterion = FocalLoss(gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = []
    best_loss = float("inf")
    best_gate_f1 = -1.0

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)
        val_metrics, val_cm, val_report = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - t0

        scheduler.step(val_metrics["macro_f1"])

        status_str = "PASS" if val_metrics["all_gates_pass"] else "FAIL"
        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.5f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Gates: {status_str} (G_rec: {val_metrics['gunshot_recall']*100:.1f}%, "
            f"D_rec: {val_metrics['deforest_recall']*100:.1f}%, "
            f"H_prec: {val_metrics['human_precision']*100:.1f}%, "
            f"V_prec: {val_metrics['vehicle_precision']*100:.1f}%)"
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "gunshot_recall": val_metrics["gunshot_recall"],
            "deforest_recall": val_metrics["deforest_recall"],
            "human_precision": val_metrics["human_precision"],
            "vehicle_precision": val_metrics["vehicle_precision"],
            "all_gates_pass": val_metrics["all_gates_pass"],
            "epoch_time_s": epoch_time,
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(history_entry)

        # 1. Save per-epoch snapshot
        snapshot_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
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
            "parameters": sum(p.numel() for p in model.parameters()),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "all_gates_pass": val_metrics["all_gates_pass"]
        }, snapshot_path)

        # 2. Save best validation loss checkpoint
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_loss_path = out_dir / "wildguard_audio_cnn_best.pt"
            torch.save({
                "epoch": epoch,
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
                "parameters": sum(p.numel() for p in model.parameters()),
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "all_gates_pass": val_metrics["all_gates_pass"]
            }, best_loss_path)
            print(f"  --> Saved new best loss checkpoint: {best_loss_path.name} (val_loss={best_loss:.4f})")

        # 3. Save best gate-compliant Macro F1 checkpoint
        if val_metrics["all_gates_pass"] and val_metrics["macro_f1"] > best_gate_f1:
            best_gate_f1 = val_metrics["macro_f1"]
            best_gates_path = out_dir / "wildguard_audio_cnn_best_gates.pt"
            torch.save({
                "epoch": epoch,
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
                "parameters": sum(p.numel() for p in model.parameters()),
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "all_gates_pass": True
            }, best_gates_path)
            print(f"  --> Saved new BEST GATES checkpoint: {best_gates_path.name} (val_macro_f1={best_gate_f1:.5f})")

    # Save latest checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "architecture": "HybridCNN",
        "parameters": sum(p.numel() for p in model.parameters())
    }, out_dir / "checkpoint_latest.pt")

    # Save training history and configuration
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE.")
    print(f"  Best Validation Loss: {best_loss:.4f}")
    print(f"  Best Gate-Compliant Macro F1: {best_gate_f1:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
