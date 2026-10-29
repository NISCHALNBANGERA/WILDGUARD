"""
Verify Reference Metrics for Baseline and E3.1 on Cleaned Manifest v16 (8,210 samples).
Outputs full classification report, confusion matrix, gates check, and saves
outputs/e3_1_hybrid_pooling/reference_metrics_v16.json.
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

from train_wildguard_audio import SmallCNN, AudioDataset, CLASSES
from train_e3_hybrid import HybridCNN

def evaluate_model(model, loader, device):
    all_preds = []
    all_targets = []
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
    manifest_path = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    print(f"Verified Cleaned Validation Partition: {len(val_df)} samples")
    print(val_df["wildguard_class"].value_counts())
    
    loader = DataLoader(AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=2)
    
    # 1. Baseline
    b_ckpt_path = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
    b_ckpt = torch.load(b_ckpt_path, map_location=device)
    b_model = SmallCNN(n_classes=6).to(device)
    b_model.load_state_dict(b_ckpt["model_state_dict"])
    b_model.eval()
    b_preds, targets = evaluate_model(b_model, loader, device)
    b_rep, b_cm, b_gates = compute_all_metrics(targets, b_preds)
    
    # 2. E3.1
    e_ckpt_path = ROOT / "outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt"
    e_ckpt = torch.load(e_ckpt_path, map_location=device)
    e_model = HybridCNN(n_classes=6).to(device)
    e_model.load_state_dict(e_ckpt["model_state_dict"])
    e_model.eval()
    e_preds, _ = evaluate_model(e_model, loader, device)
    e_rep, e_cm, e_gates = compute_all_metrics(targets, e_preds)
    
    # Print side by side
    print("\n==========================================================================================")
    print(f"OFFICIAL BENCHMARK: Baseline vs E3.1 on Verified Cleaned Validation Split ({len(val_df)} samples)")
    print("==========================================================================================")
    print(f"{'Class / Metric':<22} | {'Baseline (SmallCNN)':<28} | {'E3.1 (HybridCNN)':<28}")
    print(f"{'':<22} | {'Prec     Rec      F1     Sup':<28} | {'Prec     Rec      F1     Sup':<28}")
    print("-" * 84)
    for c in CLASSES:
        bp, br, bf, bs = b_rep[c]["precision"]*100, b_rep[c]["recall"]*100, b_rep[c]["f1-score"], int(b_rep[c]["support"])
        ep, er, ef, es = e_rep[c]["precision"]*100, e_rep[c]["recall"]*100, e_rep[c]["f1-score"], int(e_rep[c]["support"])
        print(f"{c:<22} | {bp:5.1f}%  {br:5.1f}%  {bf:.4f}  {bs:<5} | {ep:5.1f}%  {er:5.1f}%  {ef:.4f}  {es:<5}")
        
    print("-" * 84)
    print(f"{'Accuracy':<22} | {b_rep['accuracy']*100:.2f}%{'':<20} | {e_rep['accuracy']*100:.2f}%")
    print(f"{'Macro F1':<22} | {b_rep['macro avg']['f1-score']:.5f}{'':<19} | {e_rep['macro avg']['f1-score']:.5f}")
    print(f"{'Weighted F1':<22} | {b_rep['weighted avg']['f1-score']:.5f}{'':<19} | {e_rep['weighted avg']['f1-score']:.5f}")
    
    print("\n--- Regression Gates Evaluation ---")
    for g, info in e_gates.items():
        b_val = b_gates[g]["value"]
        status = "PASS" if info["pass"] else "FAIL"
        print(f"  {g:<24}: Baseline={b_val*100:.2f}%, E3.1={info['value']*100:.2f}% (Target {info['target']}) -> [{status}]")
        
    out_dict = {
        "manifest": str(manifest_path),
        "manifest_sha256": "00fed8f29e44cfa29d070c8eb11cc69c8c18084e3ceec675d406ce3ddf55dd50",
        "validation_samples": len(val_df),
        "baseline": {"rep": b_rep, "cm": b_cm.tolist(), "gates": b_gates},
        "e3_1": {"rep": e_rep, "cm": e_cm.tolist(), "gates": e_gates},
    }
    out_file = ROOT / "outputs/e3_1_hybrid_pooling/reference_metrics_v16.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved reference metrics to: {out_file}")

if __name__ == "__main__":
    main()
