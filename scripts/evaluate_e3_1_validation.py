"""
WILDGUARD E3.1 Validation Evaluator
Evaluates wildguard_audio_cnn_best.pt on the 8,213 validation samples of master_manifest_v15_e2_augmented.csv.
Exports confusion matrix CSV and detailed classification report JSON.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD"
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from scripts.train_e3_hybrid import HybridCNN, CLASSES, AudioDataset
from scripts.train_wildguard_audio import LABEL_TO_ID, SR, DURATION, N_MELS, N_FFT, HOP

def main():
    manifest_path = "data/processed/master_manifest_v15_e2_augmented.csv"
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "validation"].copy()
    print(f"Validation samples: {len(val_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_ckpt_path = "outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {best_ckpt_path} (Epoch: {ckpt.get('epoch')})")

    model = HybridCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = AudioDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    report = classification_report(all_targets, all_preds, target_names=CLASSES, output_dict=True, digits=4)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASSES))))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])

    out_dir = "outputs/e3_1_hybrid_pooling"
    cm_csv_path = os.path.join(out_dir, "validation_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to: {cm_csv_path}")

    report_json_path = os.path.join(out_dir, "validation_classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report JSON to: {report_json_path}")

    print("\n--- Validation Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=CLASSES, digits=4))

    print("\n--- Confusion Matrix ---")
    print(cm_df)

if __name__ == "__main__":
    main()
