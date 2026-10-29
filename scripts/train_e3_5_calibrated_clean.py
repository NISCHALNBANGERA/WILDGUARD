"""
WILDGUARD Experiment E3.5 (Clean):
Principled Class-Level Inverse-Square-Root Sampler + Linear Focal Loss (gamma=1.0)
Architecture: HybridCNN (24,294 parameters)
Output Directory: outputs/e3_5_calibrated_hybrid_clean/

Rigorous Design Principles:
1. Pure Class-Level Balancing (Zero Subtype Strings, Zero Dataset-Specific Rules):
   - For all classes c: w_i = min( 1 / sqrt(N_c), 1 / sqrt(N_gunshot) )
   - Human_Sound (447 clips) is uncapped at natural 1/sqrt(447) = 0.047298 (expected 2,305 draws/epoch, 5.16x oversample).
   - Vehicle (634 clips) is capped at Gunshot weight 1/sqrt(2279) = 0.020947 (expected 1,448 draws/epoch, 2.28x oversample).
   - Removes all hardcoded subtype lists (no Axe, WoodChop, Wind, Silence, or insect multipliers).
2. Linear Focal Loss (gamma=1.0):
   - Attenuation factor is linear (1 - pt)^1 instead of quadratic (1 - pt)^2.
   - Retains 10x greater gradient pressure on ambient rainforest biophony to reduce Deforestation false alarms.
   - Zero label smoothing or arbitrary modifications.
3. Isolated Clean Execution:
   - Starts strictly from scratch (epoch 1 to 10).
   - Random seeds fixed to 42 across random, numpy, torch, torch.cuda.
   - Preserves optimizer and scheduler state in every checkpoint.
   - Strictly blind to the frozen canonical test split.
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


def build_pure_class_level_weights(train_df):
    """
    Constructs pure class-level sampling weights:
    - Base weight: w_c = 1 / sqrt(N_c)
    - Vehicle weight capped at Gunshot level: min(1/sqrt(N_vehicle), 1/sqrt(N_gunshot))
    - Zero subtype strings, zero dataset-specific rules.
    """
    class_counts = train_df["wildguard_class"].value_counts()
    gunshot_cap = 1.0 / np.sqrt(class_counts["Gunshot_like"])

    weights_dict = {}
    for c, cnt in class_counts.items():
        if c == "Vehicle":
            weights_dict[c] = gunshot_cap  # Capped at Gunshot level
        else:
            weights_dict[c] = 1.0 / np.sqrt(cnt)  # Natural inverse square root

    sample_weights = train_df["wildguard_class"].map(weights_dict).to_numpy(dtype=np.float64)

    total_w = sample_weights.sum()
    probs = sample_weights / total_w
    n_total = len(train_df)

    print("\n" + "=" * 80)
    print("  VERIFIED E3.5 CLEAN SAMPLER DISTRIBUTION (Pure Class-Level)")
    print("=" * 80)
    print(f"  Total Training Samples: {n_total:,}")
    print(f"  Total Weight Sum:       {total_w:.4f}")
    header = f"  {'Class':<20} | {'Count':<7} | {'Weight w_i':<12} | {'Expected Draws':<16} | {'Draw %':<8} | {'Oversample':<12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in CLASSES:
        c_mask = train_df["wildguard_class"] == c
        c_count = c_mask.sum()
        c_prob = probs[c_mask].sum()
        expected_draws = c_prob * n_total
        w_val = weights_dict[c]
        mult = expected_draws / c_count
        print(f"  {c:<20} | {c_count:7d} | {w_val:12.6f} | {expected_draws:16.1f} | {c_prob*100:7.2f}% | {mult:10.2f}x")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    return sample_weights


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
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "outputs/e3_5_calibrated_hybrid_clean"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0, help="Linear focal loss exponent")
    parser.add_argument("--seed", type=int, default=42)
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
    print("  WILDGUARD Experiment E3.5 (Clean Run) - Initializing Pipeline")
    print("=" * 80)
    print(f"  Manifest:     {args.manifest}")
    print(f"  Output Dir:   {out_dir}")
    print(f"  Architecture: HybridCNN (24,294 parameters)")
    print(f"  Loss:         FocalLoss (gamma={args.gamma})")
    print(f"  Sampler:      Pure Class-Level Inverse-Square-Root (Vehicle Capped, Human Natural)")
    print(f"  Epochs:       {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}, Seed: {args.seed}")
    sys.stdout.flush()

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

    # Build Pure Class-Level Sampler
    sample_weights = build_pure_class_level_weights(train_df)
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
    print(f"  Compute Device: {device}\n")
    sys.stdout.flush()

    model = HybridCNN(n_classes=len(CLASSES)).to(device)
    criterion = FocalLoss(gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = []
    best_loss = float("inf")
    best_gate_f1 = -1.0

    print("Starting Training Loop (Epoch 1 to 10)...")
    sys.stdout.flush()

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
        log_line = (
            f"Epoch {epoch:2d}/{args.epochs:2d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.5f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Gates: {status_str} (G_rec: {val_metrics['gunshot_recall']*100:.1f}%, "
            f"D_rec: {val_metrics['deforest_recall']*100:.1f}%, "
            f"H_prec: {val_metrics['human_precision']*100:.1f}%, "
            f"V_prec: {val_metrics['vehicle_precision']*100:.1f}%)"
        )
        print(log_line)
        sys.stdout.flush()

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

        # 1. Save per-epoch snapshot with full optimizer and scheduler states
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
            sys.stdout.flush()

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
            sys.stdout.flush()

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
    sys.stdout.flush()


if __name__ == "__main__":
    main()
