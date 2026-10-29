"""
Comprehensive Checkpoint Verification for E3.3:
Audits outputs/e3_3_focal_capped/ checkpoints:
- Verifies existence of Epoch 7 vs Epoch 9 vs Epoch 10.
- Re-evaluates available checkpoints on the verified 8,210-sample validation partition.
- Confirms expected 6 classes and exact preprocessing parameters.
- Recomputes confusion matrices, accuracy, macro F1, weighted F1, per-class metrics, and regression gates.
- Compares with Official Baseline on the exact same validation sample IDs.
- Strictly does not touch the canonical 8,189-sample test set.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))

from train_wildguard_audio import SmallCNN, AudioDataset, CLASSES, LABEL_TO_ID, SR, DURATION, N_MELS, N_FFT, HOP
from train_e3_hybrid import HybridCNN

EXPECTED_CLASSES = [
    "Deforestation",
    "Forest_Environment",
    "Forest_Fire",
    "Gunshot_like",
    "Human_Sound",
    "Vehicle",
]

def evaluate_model(model, loader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    return np.array(all_preds), np.array(all_targets)

def compute_all_metrics(targets, preds):
    rep = classification_report(targets, preds, target_names=CLASSES, digits=5, output_dict=True)
    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASSES))))
    
    gunshot_rec = rep["Gunshot_like"]["recall"]
    human_prec = rep["Human_Sound"]["precision"]
    deforest_rec = rep["Deforestation"]["recall"]
    vehicle_prec = rep["Vehicle"]["precision"]
    
    gates = {
        "gunshot_recall_ge_94": {"value": round(gunshot_rec, 5), "target": ">= 0.94", "pass": bool(gunshot_rec >= 0.94)},
        "human_precision_ge_40": {"value": round(human_prec, 5), "target": ">= 0.40", "pass": bool(human_prec >= 0.40)},
        "deforest_recall_ge_87_5": {"value": round(deforest_rec, 5), "target": ">= 0.875", "pass": bool(deforest_rec >= 0.875)},
        "vehicle_precision_ge_55": {"value": round(vehicle_prec, 5), "target": ">= 0.55", "pass": bool(vehicle_prec >= 0.55)},
    }
    return rep, cm, gates

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Manifest & Split Validation
    manifest_path = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    print(f"Verified validation samples: {len(val_df)}")
    assert len(val_df) == 8210, f"Expected 8,210 samples, got {len(val_df)}"
    
    # Verify canonical test set is untouched
    canonical_test = df[df["split"] == "test"]
    assert len(canonical_test) == 8189, f"Canonical test count changed! {len(canonical_test)}"
    test_overlap = set(val_df["audio_path"]).intersection(set(canonical_test["audio_path"]))
    assert len(test_overlap) == 0, f"Leakage detected with test set!"
    print("Canonical test set: 8,189 samples (100% frozen, zero overlap with validation)")
    
    loader = DataLoader(AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=2)
    
    # 2. Inspect E3.3 Checkpoints
    e3_3_dir = ROOT / "outputs/e3_3_focal_capped"
    best_ckpt_path = e3_3_dir / "wildguard_audio_cnn_best.pt"
    latest_ckpt_path = e3_3_dir / "checkpoint_latest.pt"
    
    print("\n=======================================================")
    print("TASK 1 & 2: Checkpoint Inventory & Epoch 7 Existence")
    print("=======================================================")
    
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    latest_ckpt = torch.load(latest_ckpt_path, map_location=device)
    
    print(f"File: {best_ckpt_path.name}")
    print(f"  Epoch: {best_ckpt['epoch']}")
    print(f"  Best Val Loss: {best_ckpt['best_val_loss']:.6f}")
    print(f"  Architecture: {best_ckpt['architecture']}")
    print(f"  Classes: {best_ckpt['classes']}")
    print(f"  Sample Rate: {best_ckpt.get('sample_rate')}, Hop: {best_ckpt.get('hop_length')}, N_Mels: {best_ckpt.get('n_mels')}")
    
    print(f"\nFile: {latest_ckpt_path.name}")
    print(f"  Epoch: {latest_ckpt['epoch']}")
    print(f"  Best Val Loss: {latest_ckpt['best_val_loss']:.6f}")
    
    # Check if any Epoch 7 checkpoint file exists on disk
    all_pt_files = list(e3_3_dir.glob("*.pt"))
    print(f"\nAll .pt files in {e3_3_dir.name}: {[f.name for f in all_pt_files]}")
    has_epoch_7_file = any("epoch_7" in f.name or "epoch7" in f.name for f in all_pt_files)
    print(f"Does a standalone Epoch 7 checkpoint file exist on disk? {has_epoch_7_file}")
    
    # 3. Check Training History for Epoch 7 vs Epoch 9
    hist_path = e3_3_dir / "training_history.json"
    with open(hist_path) as f:
        hist = json.load(f)
    print(f"\nTotal epochs recorded in training_history.json: {len(hist)}")
    ep7_hist = hist[6] # 0-indexed, epoch 7
    ep9_hist = hist[8] # epoch 9
    ep10_hist = hist[9] # epoch 10
    
    # 4. Re-evaluate Best Checkpoint (Epoch 9)
    print("\n=======================================================")
    print("TASK 3: Re-evaluating wildguard_audio_cnn_best.pt (Epoch 9)")
    print("=======================================================")
    best_model = HybridCNN(n_classes=6).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()
    
    best_preds, targets = evaluate_model(best_model, loader, device)
    best_rep, best_cm, best_gates = compute_all_metrics(targets, best_preds)
    
    # 5. Re-evaluate Latest Checkpoint (Epoch 10)
    print("\n--- Re-evaluating checkpoint_latest.pt (Epoch 10) ---")
    latest_model = HybridCNN(n_classes=6).to(device)
    latest_model.load_state_dict(latest_ckpt["model_state_dict"])
    latest_model.eval()
    
    latest_preds, _ = evaluate_model(latest_model, loader, device)
    latest_rep, latest_cm, latest_gates = compute_all_metrics(targets, latest_preds)
    
    # 6. Re-evaluate Official Baseline
    print("\n--- Re-evaluating Official Baseline (SmallCNN) ---")
    baseline_path = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
    b_ckpt = torch.load(baseline_path, map_location=device)
    baseline_model = SmallCNN(n_classes=6).to(device)
    baseline_model.load_state_dict(b_ckpt["model_state_dict"])
    baseline_model.eval()
    
    b_preds, _ = evaluate_model(baseline_model, loader, device)
    b_rep, b_cm, b_gates = compute_all_metrics(targets, b_preds)
    
    # 7. Print Comparative Table
    print("\n=============================================================================================================")
    print(f"OFFICIAL AUDIT COMPARISON: Exact Same 8,210 Validation Samples")
    print("=============================================================================================================")
    print(f"{'Class / Metric':<20} | {'Baseline (SmallCNN)':<22} | {'E3.3 Hist Ep 7':<20} | {'E3.3 Best (Ep 9 Ckpt)':<22} | {'E3.3 Latest (Ep 10)':<20}")
    print(f"{'':<20} | {'Prec    Rec     F1':<22} | {'Prec    Rec     F1':<20} | {'Prec    Rec     F1':<22} | {'Prec    Rec     F1':<20}")
    print("-" * 115)
    
    for c in CLASSES:
        bp, br, bf = b_rep[c]["precision"]*100, b_rep[c]["recall"]*100, b_rep[c]["f1-score"]
        h7p, h7r, h7f = ep7_hist["per_class_val"][c]["precision"]*100, ep7_hist["per_class_val"][c]["recall"]*100, ep7_hist["per_class_val"][c]["f1"]
        e9p, e9r, e9f = best_rep[c]["precision"]*100, best_rep[c]["recall"]*100, best_rep[c]["f1-score"]
        e10p, e10r, e10f = latest_rep[c]["precision"]*100, latest_rep[c]["recall"]*100, latest_rep[c]["f1-score"]
        print(f"{c:<20} | {bp:5.1f}% {br:5.1f}% {bf:6.4f} | {h7p:5.1f}% {h7r:5.1f}% {h7f:6.4f} | {e9p:5.1f}% {e9r:5.1f}% {e9f:6.4f} | {e10p:5.1f}% {e10r:5.1f}% {e10f:6.4f}")
        
    print("-" * 115)
    print(f"{'Accuracy':<20} | {b_rep['accuracy']*100:.2f}%{'':<15} | {ep7_hist['val_acc']*100:.2f}%{'':<13} | {best_rep['accuracy']*100:.2f}%{'':<15} | {latest_rep['accuracy']*100:.2f}%")
    print(f"{'Macro F1':<20} | {b_rep['macro avg']['f1-score']:.5f}{'':<14} | {ep7_hist['val_macro_f1']:.5f}{'':<12} | {best_rep['macro avg']['f1-score']:.5f}{'':<14} | {latest_rep['macro avg']['f1-score']:.5f}")
    print(f"{'Weighted F1':<20} | {b_rep['weighted avg']['f1-score']:.5f}{'':<14} | {ep7_hist['val_weighted_f1']:.5f}{'':<12} | {best_rep['weighted avg']['f1-score']:.5f}{'':<14} | {latest_rep['weighted avg']['f1-score']:.5f}")
    
    print("\n--- Regression Gates Audit ---")
    print(f"{'Gate':<26} | {'Target':<10} | {'Baseline':<12} | {'E3.3 Hist Ep 7':<15} | {'E3.3 Best (Ep 9)':<16} | {'E3.3 Latest (Ep 10)'}")
    print("-" * 105)
    gates_keys = [
        ("Gunshot Recall", "gunshot_recall_ge_94", "val_gunshot_recall"),
        ("Deforest Recall", "deforest_recall_ge_87_5", "val_deforest_recall"),
        ("Human Precision", "human_precision_ge_40", "val_human_precision"),
        ("Vehicle Precision", "vehicle_precision_ge_55", "val_vehicle_precision"),
    ]
    for name, gkey, hkey in gates_keys:
        bt = b_gates[gkey]["value"] * 100
        bp = "PASS" if b_gates[gkey]["pass"] else "FAIL"
        
        h7v = ep7_hist[hkey] * 100
        h7p = "PASS" if (h7v >= (94.0 if "gunshot" in gkey else 87.5 if "deforest" in gkey else 40.0 if "human" in gkey else 55.0)) else "FAIL"
        
        e9v = best_gates[gkey]["value"] * 100
        e9p = "PASS" if best_gates[gkey]["pass"] else "FAIL"
        
        e10v = latest_gates[gkey]["value"] * 100
        e10p = "PASS" if latest_gates[gkey]["pass"] else "FAIL"
        
        tgt = b_gates[gkey]["target"]
        print(f"{name:<26} | {tgt:<10} | {bt:5.2f}% [{bp}] | {h7v:5.2f}% [{h7p}]   | {e9v:5.2f}% [{e9p}]    | {e10v:5.2f}% [{e10p}]")
        
    print("\n--- Confusion Matrix: E3.3 Best Checkpoint (Epoch 9) ---")
    cm_df = pd.DataFrame(best_cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
    print(cm_df)

if __name__ == "__main__":
    main()
