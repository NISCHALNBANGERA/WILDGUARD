"""
Generate E3.2 Evaluation Artifacts on Cleaned Manifest v16 (8,210 samples).
Exports confusion matrix CSV, detailed report JSON, and markdown summary.
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
    manifest_path = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    loader = DataLoader(AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=2)
    
    ckpt_path = ROOT / "outputs/e3_2_focal_loss/wildguard_audio_cnn_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
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
    
    out_dir = ROOT / "outputs/e3_2_focal_loss"
    cm_csv_path = out_dir / "e3_2_validation_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to: {cm_csv_path}")
    
    rep_json_path = out_dir / "e3_2_validation_report.json"
    with open(rep_json_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Saved report JSON to: {rep_json_path}")
    
    print("\n--- E3.2 Confusion Matrix ---")
    print(cm_df)

if __name__ == "__main__":
    main()
