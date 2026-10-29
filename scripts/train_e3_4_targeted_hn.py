"""
WILDGUARD Experiment E3.4 Training Pipeline:
Targeted Hard-Negative Balanced Sampler + Focal Loss (gamma=2.0)
Architecture: HybridCNN (24,294 parameters)
Manifest: data/processed/master_manifest_v16_e3_cleaned.csv (8,210 leak-free validation samples)

Key Innovations in E3.4:
1. Vehicle Weight Capped: Caps Vehicle sampling weight at Gunshot level (0.020947), eliminating the 53x sampling inflation that caused vehicle false alarms on weather rumbles.
2. Ambient Rumble Hard-Negative Boost: Applies a 2.5x sampling multiplier to low-frequency weather subtypes (thunderstorm, wind, sea_waves, rain, silence) within Forest_Environment, teaching the network to distinguish natural rumble from engine hum.
3. Manual Logging Boost: Preserves 1.5x multiplier on Axe, WoodChop, TreeFalling, and Handsaw strikes.
4. Focal Loss (gamma=2.0): Eliminates ambient rainforest gradient dominance on distant RFCx chainsaws.
5. Per-Epoch Snapshotting: Saves checkpoint_epoch_{epoch}.pt at every epoch, plus wildguard_audio_cnn_best_gates.pt (best gate-compliant Macro F1) and wildguard_audio_cnn_best.pt (best validation loss).
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

def build_targeted_hard_negative_weights(train_df):
    """
    Constructs targeted hard-negative sampling weights:
    - Base weights: 1/sqrt(class_count)
    - Gunshot weight is the universal minority cap (prevents prior explosion)
    - CAP Human_Sound at gunshot level
    - CAP Vehicle at gunshot level (prevents the 53x prior inflation from E3.1/E3.3)
    - Boost ambient weather rumble hard-negatives in Forest_Environment by 2.5x
    - Boost manual logging minority threat subtypes by 1.5x
    """
    class_counts = train_df["wildguard_class"].value_counts()
    base_weights = train_df["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    gunshot_cap = 1.0 / np.sqrt(class_counts["Gunshot_like"])
    sample_weights = base_weights.copy()

    # 1. Cap Human_Sound at gunshot level
    human_mask = train_df["wildguard_class"] == "Human_Sound"
    orig_human = sample_weights[human_mask].iloc[0]
    sample_weights[human_mask] = gunshot_cap

    # 2. Cap Vehicle at gunshot level (replaces unconstrained 1.5x inflation)
    vehicle_mask = train_df["wildguard_class"] == "Vehicle"
    orig_vehicle = sample_weights[vehicle_mask].iloc[0]
    sample_weights[vehicle_mask] = gunshot_cap

    # 3. Boost ambient weather rumble in Forest_Environment
    rumble_mask = (
        (train_df["wildguard_class"] == "Forest_Environment") &
        (train_df["source_label"].astype(str).str.lower().isin(RUMBLE_HARD_NEGATIVES))
    )
    sample_weights[rumble_mask] *= 2.5

    # 4. Boost manual logging subtypes in Deforestation
    logging_mask = (
        (train_df["wildguard_class"] == "Deforestation") &
        (train_df["source_label"].isin(DEFOREST_MINORITY_SUBTYPES))
    )
    sample_weights[logging_mask] *= 1.5

    non_rumble_mask = (train_df["wildguard_class"] == "Forest_Environment") & (~rumble_mask)
    forest_base_weight = sample_weights[non_rumble_mask].iloc[0]
    forest_rumble_weight = sample_weights[rumble_mask].iloc[0]
    print(f"\n  === Targeted Hard-Negative Sampler Audit ===")
    print(f"    Forest_Environment base weight:   {forest_base_weight:.6f}")
    print(f"    Forest_Environment RUMBLE weight: {forest_rumble_weight:.6f} (2.5x hard-negative boost)")
    print(f"    Deforestation base weight:        {sample_weights[train_df['wildguard_class']=='Deforestation'].iloc[0]:.6f}")
    print(f"    Gunshot_like cap weight:          {gunshot_cap:.6f}")
    print(f"    Human_Sound capped weight:        {sample_weights[human_mask].iloc[0]:.6f} (was {orig_human:.6f})")
    print(f"    Vehicle CAPPED weight:            {sample_weights[vehicle_mask].iloc[0]:.6f} (was {orig_vehicle:.6f})")
    print(f"    Vehicle / Forest_Env base ratio:  {sample_weights[vehicle_mask].iloc[0] / forest_base_weight:.2f}x (was 8.7x in E3.3)")
    print(f"    Rumble hard-negative samples:     {rumble_mask.sum():,}")
    print(f"    Manual logging boosted samples:   {logging_mask.sum():,}")

    return sample_weights.to_numpy(dtype=np.float64)

def parse_args():
    parser = argparse.ArgumentParser(description="WILDGUARD E3.4 Targeted Training Pipeline")
    parser.add_argument("--experiment", type=str, default="e3_4_targeted_hn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else ROOT / f"outputs/{args.experiment}"

    best_loss_path = out_dir / "wildguard_audio_cnn_best.pt"
    best_gates_path = out_dir / "wildguard_audio_cnn_best_gates.pt"
    latest_path = out_dir / "checkpoint_latest.pt"
    history_path = out_dir / "training_history.json"
    config_path = out_dir / "experiment_config.json"

    if best_loss_path.exists() and not args.resume and not args.smoke_test:
        print(f"TERMINATING: Output folder '{out_dir}' already contains a checkpoint.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load manifest
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

    # 2. Instantiate Model
    model = HybridCNN(n_classes=len(CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridCNN (total params: {total_params:,})")
    assert total_params == 24294, f"Parameter count mismatch: {total_params}"

    sample_x, sample_y = AudioDataset(train.head(1))[0]
    with torch.no_grad():
        out = model(sample_x.unsqueeze(0).to(device))
    print(f"Input feature shape: {tuple(sample_x.shape)} | Logits output: {tuple(out.shape)}")
    assert out.shape == (1, 6), f"Output shape mismatch: {out.shape}"

    if args.smoke_test:
        print("\nSMOKE TEST COMPLETED SUCCESSFULLY.")
        return

    # 3. Setup DataLoader with Targeted Hard-Negative Sampler
    print("\nConfiguring targeted_hard_negative sampler...")
    weights = build_targeted_hard_negative_weights(train)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train),
        replacement=True
    )
    train_loader = DataLoader(
        AudioDataset(train), batch_size=args.batch_size,
        sampler=sampler, num_workers=args.workers,
    )
    val_loader = DataLoader(
        AudioDataset(val), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    # 4. Setup Loss & Optimizer
    print(f"Loss: FocalLoss (gamma={args.gamma})")
    criterion = FocalLoss(gamma=args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = []
    best_val_loss = float("inf")
    best_gates_f1 = 0.0
    start_epoch = 1

    if args.resume and latest_path.exists():
        print(f"Resuming training from {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history = ckpt.get("history", [])

    def evaluate(loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        val_rep = classification_report(all_targets, all_preds, target_names=CLASSES, digits=5, output_dict=True)
        val_cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
        return total_loss / total, val_rep, val_cm

    print("\n" + "=" * 70)
    print(f"Starting Training: {args.experiment} (Epochs {start_epoch} to {args.epochs})")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

            if batch_idx % 250 == 0 or batch_idx == len(train_loader):
                cur_train_loss = train_loss / train_total
                cur_train_acc = train_correct / train_total
                print(f"  [Epoch {epoch} | Step {batch_idx}/{len(train_loader)}] loss={cur_train_loss:.4f} acc={cur_train_acc:.4f}", flush=True)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        val_loss, val_rep, val_cm = evaluate(val_loader)
        scheduler.step(val_loss)

        val_acc = val_rep["accuracy"]
        val_macro_f1 = val_rep["macro avg"]["f1-score"]
        val_weighted_f1 = val_rep["weighted avg"]["f1-score"]
        lr = optimizer.param_groups[0]["lr"]

        val_gunshot_rec = val_rep["Gunshot_like"]["recall"]
        val_deforest_rec = val_rep["Deforestation"]["recall"]
        val_human_prec = val_rep["Human_Sound"]["precision"]
        val_vehicle_prec = val_rep["Vehicle"]["precision"]

        gates_pass = (
            (val_gunshot_rec >= 0.94) and
            (val_deforest_rec >= 0.875) and
            (val_human_prec >= 0.40) and
            (val_vehicle_prec >= 0.55)
        )

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
        print(f"      All Gates Passed?    {'*** ALL 4 PASS ***' if gates_pass else 'NO'}")

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
            "all_gates_pass": bool(gates_pass),
            "lr": lr,
            "per_class_val": {
                c: {
                    "precision": round(val_rep[c]["precision"], 4),
                    "recall": round(val_rep[c]["recall"], 4),
                    "f1": round(val_rep[c]["f1-score"], 4),
                }
                for c in CLASSES
            },
        }
        history.append(epoch_stats)

        # 1. Save per-epoch checkpoint snapshot
        epoch_ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        ckpt_data = {
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
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "all_gates_pass": bool(gates_pass),
        }
        torch.save(ckpt_data, epoch_ckpt_path)
        torch.save(ckpt_data, latest_path)

        # 2. Track & save BEST GATE-COMPLIANT Checkpoint
        if gates_pass and val_macro_f1 > best_gates_f1:
            best_gates_f1 = val_macro_f1
            torch.save(ckpt_data, best_gates_path)
            print(f"  *** NEW BEST GATE-COMPLIANT CHECKPOINT SAVED: Epoch {epoch} (macro_f1={val_macro_f1:.4f}, all 4 gates pass) ***")

        # 3. Track & save BEST LOSS Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, best_loss_path)
            print(f"  *** NEW BEST LOSS CHECKPOINT SAVED: Epoch {epoch} (val_loss={val_loss:.4f}) ***")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Save final config
    exp_config = {
        "experiment": args.experiment,
        "manifest": str(manifest_path),
        "manifest_sha256": "00fed8f29e44cfa29d070c8eb11cc69c8c18084e3ceec675d406ce3ddf55dd50",
        "architecture": "HybridCNN",
        "parameters": total_params,
        "classes": CLASSES,
        "class_index": LABEL_TO_ID,
        "preprocessing": {"sample_rate": SR, "duration_seconds": DURATION, "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "loss": "focal", "gamma": args.gamma, "sampler": "targeted_hard_negative"},
        "best_gates_macro_f1": best_gates_f1,
        "best_val_loss": best_val_loss,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED.")
    print(f"Best Loss Checkpoint:          {best_loss_path} (val_loss={best_val_loss:.4f})")
    if best_gates_path.exists():
        print(f"Best Gate-Compliant Checkpoint: {best_gates_path} (macro_f1={best_gates_f1:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
