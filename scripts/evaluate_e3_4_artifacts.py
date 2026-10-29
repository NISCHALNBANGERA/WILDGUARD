"""
WILDGUARD E3.4 Artifact Generator & Evaluator:
Evaluates wildguard_audio_cnn_best_gates.pt (Epoch 9) on the verified 8,210 validation partition.
Computes confusion matrix, full classification report, per-class metrics, and gates verification.
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

from train_wildguard_audio import AudioDataset, CLASSES
from train_e3_hybrid import HybridCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    manifest_path = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    assert len(val_df) == 8210, f"Expected 8,210 samples, got {len(val_df)}"
    print(f"Verified Validation Samples: {len(val_df)}")
    
    loader = DataLoader(AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=2)
    
    ckpt_path = ROOT / "outputs/e3_4_targeted_hn/wildguard_audio_cnn_best_gates.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Val Macro F1 in ckpt: {ckpt.get('val_macro_f1'):.5f}")
    print(f"  Val Loss in ckpt: {ckpt.get('val_loss'):.5f}")
    
    model = HybridCNN(n_classes=6).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rep = classification_report(all_targets, all_preds, target_names=CLASSES, digits=5, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
    
    out_dir = ROOT / "outputs/e3_4_targeted_hn"
    cm_csv = out_dir / "e3_4_validation_confusion_matrix.csv"
    cm_df.to_csv(cm_csv)
    print(f"\nSaved confusion matrix to: {cm_csv}")
    
    rep_json = out_dir / "e3_4_validation_report.json"
    with open(rep_json, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Saved classification report to: {rep_json}")
    
    print("\n=======================================================")
    print("INDEPENDENT RE-COMPUTED METRICS: wildguard_audio_cnn_best_gates.pt")
    print("=======================================================")
    print(f"Accuracy:    {rep['accuracy']*100:.2f}%")
    print(f"Macro F1:    {rep['macro avg']['f1-score']:.5f}")
    print(f"Weighted F1: {rep['weighted avg']['f1-score']:.5f}")
    print("-" * 65)
    for c in CLASSES:
        p = rep[c]["precision"] * 100
        r = rep[c]["recall"] * 100
        f1 = rep[c]["f1-score"]
        sup = int(rep[c]["support"])
        print(f"  {c:<20s}: Precision={p:5.1f}%, Recall={r:5.1f}%, F1={f1:.4f}, Support={sup}")
        
    print("\n--- Regression Gates Audit ---")
    gun_rec = rep["Gunshot_like"]["recall"] * 100
    def_rec = rep["Deforestation"]["recall"] * 100
    hum_prec = rep["Human_Sound"]["precision"] * 100
    veh_prec = rep["Vehicle"]["precision"] * 100
    
    print(f"  Gunshot Recall:    {gun_rec:5.2f}% (Target >= 94.0%: {'PASS' if gun_rec>=94.0 else 'FAIL'})")
    print(f"  Deforest Recall:   {def_rec:5.2f}% (Target >= 87.5%: {'PASS' if def_rec>=87.5 else 'FAIL'})")
    print(f"  Human Precision:   {hum_prec:5.2f}% (Target >= 40.0%: {'PASS' if hum_prec>=40.0 else 'FAIL'})")
    print(f"  Vehicle Precision: {veh_prec:5.2f}% (Target >= 55.0%: {'PASS' if veh_prec>=55.0 else 'FAIL'})")
    
    print("\n--- Confusion Matrix ---")
    print(cm_df)

if __name__ == "__main__":
    main()
