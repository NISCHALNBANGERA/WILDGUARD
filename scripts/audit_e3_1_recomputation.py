"""
Diagnostic Audit: Recompute E3.1 validation metrics from the verified 8,213 partition
and compare with the 8,134 partition.
Preserves all checkpoints, does not evaluate the frozen canonical test set.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD"
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from scripts.train_e3_hybrid import HybridCNN, CLASSES, AudioDataset

def run_eval(df_subset, name, model, device):
    print(f"\n=======================================================")
    print(f"Evaluating: {name} ({len(df_subset)} samples)")
    print(f"=======================================================")
    print("Class distribution:")
    print(df_subset["wildguard_class"].value_counts().to_string())
    
    dataset = AudioDataset(df_subset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rep = classification_report(all_targets, all_preds, target_names=CLASSES, digits=5, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
    
    print("\n--- Per-Class Results ---")
    for c in CLASSES:
        p = rep[c]["precision"] * 100
        r = rep[c]["recall"] * 100
        f1 = rep[c]["f1-score"]
        sup = rep[c]["support"]
        print(f"  {c:20s}: Precision={p:6.2f}%, Recall={r:6.2f}%, F1={f1:.4f}, Support={sup:5.0f}")
        
    print(f"\nAccuracy:    {rep['accuracy']*100:.2f}%")
    print(f"Macro F1:    {rep['macro avg']['f1-score']:.5f}")
    print(f"Weighted F1: {rep['weighted avg']['f1-score']:.5f}")
    
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())
    return rep, cm_df

def main():
    manifest_path = "data/processed/master_manifest_v15_e2_augmented.csv"
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    # 1. Verified 8,213 partition (used in train_e3_hybrid.py)
    val_8213 = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    # 2. Legacy 8,134 partition (used in evaluate_e3_1_validation.py)
    val_8134 = df[df["split"] == "validation"].copy().reset_index(drop=True)
    
    # Load checkpoint
    ckpt_path = "outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint Epoch: {ckpt.get('epoch')}")
    print(f"Checkpoint Best Val Loss: {ckpt.get('best_val_loss')}")
    
    model = HybridCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Run evaluations
    rep_8213, cm_8213 = run_eval(val_8213, "Verified Validation Split (train_e3_hybrid.py)", model, device)
    rep_8134, cm_8134 = run_eval(val_8134, "Legacy Validation Filter (evaluate_e3_1_validation.py)", model, device)
    
    # Compare with training_history.json
    hist_path = "outputs/e3_1_hybrid_pooling/training_history.json"
    with open(hist_path) as f:
        hist = json.load(f)
    ep10 = hist[-1]
    
    print("\n=======================================================")
    print("Reconciliation Audit vs training_history.json (Epoch 10)")
    print("=======================================================")
    print(f"Macro F1 in training_history.json: {ep10['val_macro_f1']:.5f}")
    print(f"Macro F1 on 8,213 partition:       {rep_8213['macro avg']['f1-score']:.5f}")
    print(f"Macro F1 on 8,134 partition:       {rep_8134['macro avg']['f1-score']:.5f}")
    print(f"Difference (8,213 vs history):     {abs(rep_8213['macro avg']['f1-score'] - ep10['val_macro_f1']):.7f}")

if __name__ == "__main__":
    main()
