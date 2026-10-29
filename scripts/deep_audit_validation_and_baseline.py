"""
Deep Audit: Recomputes Baseline vs E3.1 on Verified 8,213 Validation Partition.
Traces Deforestation -> Forest_Environment and Forest_Environment -> Vehicle errors.
Audits parent group split isolation.
Preserves all checkpoints, does not touch canonical test set.
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

from train_wildguard_audio import (
    SmallCNN,
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
from train_e3_hybrid import HybridCNN

def evaluate_model(model, loader, device):
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_probs.extend(probs)
            
    return np.array(all_preds), np.array(all_targets), np.array(all_probs)

def compute_metrics(targets, preds):
    rep = classification_report(targets, preds, target_names=CLASSES, digits=5, output_dict=True)
    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASSES))))
    
    # Regression gates
    gunshot_rec = rep["Gunshot_like"]["recall"]
    human_prec = rep["Human_Sound"]["precision"]
    deforest_rec = rep["Deforestation"]["recall"]
    vehicle_prec = rep["Vehicle"]["precision"]
    
    gates = {
        "gunshot_recall_ge_94": {"value": gunshot_rec, "pass": bool(gunshot_rec >= 0.94)},
        "human_precision_ge_40": {"value": human_prec, "pass": bool(human_prec >= 0.40)},
        "deforest_recall_ge_87_5": {"value": deforest_rec, "pass": bool(deforest_rec >= 0.875)},
        "vehicle_precision_ge_55": {"value": vehicle_prec, "pass": bool(vehicle_prec >= 0.55)},
    }
    
    return rep, cm, gates

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load manifest and select verified 8,213 validation partition
    manifest_path = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)
    
    print(f"Verified validation samples: {len(val_df)}")
    print(val_df["wildguard_class"].value_counts())
    
    val_loader = DataLoader(
        AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=2
    )
    
    # 2. Evaluate Baseline Model
    baseline_ckpt_path = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
    print(f"\n--- Loading Baseline Checkpoint: {baseline_ckpt_path} ---")
    b_ckpt = torch.load(baseline_ckpt_path, map_location=device)
    baseline_model = SmallCNN(n_classes=len(CLASSES)).to(device)
    baseline_model.load_state_dict(b_ckpt["model_state_dict"])
    baseline_model.eval()
    
    b_preds, b_targets, b_probs = evaluate_model(baseline_model, val_loader, device)
    b_rep, b_cm, b_gates = compute_metrics(b_targets, b_preds)
    
    # 3. Evaluate E3.1 HybridCNN
    e3_1_ckpt_path = ROOT / "outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt"
    print(f"\n--- Loading E3.1 Checkpoint: {e3_1_ckpt_path} ---")
    e_ckpt = torch.load(e3_1_ckpt_path, map_location=device)
    e3_1_model = HybridCNN(n_classes=len(CLASSES)).to(device)
    e3_1_model.load_state_dict(e_ckpt["model_state_dict"])
    e3_1_model.eval()
    
    e_preds, e_targets, e_probs = evaluate_model(e3_1_model, val_loader, device)
    e_rep, e_cm, e_gates = compute_metrics(e_targets, e_preds)
    
    # Add predictions to dataframe for error tracing
    val_df["baseline_pred"] = [CLASSES[p] for p in b_preds]
    val_df["e3_1_pred"] = [CLASSES[p] for p in e_preds]
    val_df["baseline_correct"] = val_df["wildguard_class"] == val_df["baseline_pred"]
    val_df["e3_1_correct"] = val_df["wildguard_class"] == val_df["e3_1_pred"]
    
    # 4. Error Tracing
    # Deforestation -> Forest_Environment
    deforest_to_env = val_df[
        (val_df["wildguard_class"] == "Deforestation") &
        (val_df["e3_1_pred"] == "Forest_Environment")
    ]
    print(f"\nTotal Deforestation -> Forest_Environment errors: {len(deforest_to_env)}")
    print("Breakdown by dataset:")
    print(deforest_to_env["dataset"].value_counts())
    print("\nBreakdown by source_label:")
    print(deforest_to_env["source_label"].value_counts().head(10))
        
    # Manual logging inspection
    manual_subtypes = ["Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw"]
    val_manual = val_df[val_df["source_label"].isin(manual_subtypes)]
    print(f"\nManual logging samples in validation: {len(val_manual)}")
    if len(val_manual) > 0:
        for st in manual_subtypes:
            st_df = val_df[val_df["source_label"] == st]
            if len(st_df) > 0:
                b_acc = (st_df["baseline_pred"] == "Deforestation").mean()
                e_acc = (st_df["e3_1_pred"] == "Deforestation").mean()
                print(f"  {st:15s} (N={len(st_df)}): Baseline Recall={b_acc*100:.1f}%, E3.1 Recall={e_acc*100:.1f}%")
                
    # Forest_Environment -> Vehicle
    env_to_vehicle = val_df[
        (val_df["wildguard_class"] == "Forest_Environment") &
        (val_df["e3_1_pred"] == "Vehicle")
    ]
    print(f"\nTotal Forest_Environment -> Vehicle errors: {len(env_to_vehicle)}")
    print("Breakdown by dataset:")
    print(env_to_vehicle["dataset"].value_counts())
    print("\nBreakdown by source_label:")
    print(env_to_vehicle["source_label"].value_counts().head(10))
        
    # 5. Split Isolation & Parent Group Audit
    print("\n--- Auditing Parent Group Leakage ---")
    parent_col = "source_group_id"
    leakage_found = False
    print(f"Using parent group column: {parent_col}")
    train_parents = set(df[df["split_plan_v1"] == "train"][parent_col].dropna())
    val_parents = set(df[df["split_plan_v1"] == "validation"][parent_col].dropna())
    test_parents = set(df[df["split_plan_v1"] == "test"][parent_col].dropna())
    
    train_val = train_parents.intersection(val_parents)
    train_test = train_parents.intersection(test_parents)
    val_test = val_parents.intersection(test_parents)
    
    print(f"Train parent groups:      {len(train_parents):,}")
    print(f"Validation parent groups: {len(val_parents):,}")
    print(f"Test parent groups:       {len(test_parents):,}")
    print(f"Train/Val overlap:        {len(train_val)}")
    print(f"Train/Test overlap:       {len(train_test)}")
    print(f"Val/Test overlap:         {len(val_test)}")
    if len(train_val) + len(train_test) + len(val_test) > 0:
        leakage_found = True
        
    # 6. Save Complete Comparative JSON
    comparison_summary = {
        "manifest": str(manifest_path),
        "validation_samples": len(val_df),
        "baseline": {
            "checkpoint": str(baseline_ckpt_path),
            "macro_f1": b_rep["macro avg"]["f1-score"],
            "weighted_f1": b_rep["weighted avg"]["f1-score"],
            "accuracy": b_rep["accuracy"],
            "gates": b_gates,
            "per_class": {c: b_rep[c] for c in CLASSES},
            "confusion_matrix": b_cm.tolist()
        },
        "e3_1": {
            "checkpoint": str(e3_1_ckpt_path),
            "macro_f1": e_rep["macro avg"]["f1-score"],
            "weighted_f1": e_rep["weighted avg"]["f1-score"],
            "accuracy": e_rep["accuracy"],
            "gates": e_gates,
            "per_class": {c: e_rep[c] for c in CLASSES},
            "confusion_matrix": e_cm.tolist()
        },
        "errors": {
            "deforest_to_env_count": len(deforest_to_env),
            "deforest_to_env_by_dataset": deforest_to_env["dataset"].value_counts().to_dict(),
            "env_to_vehicle_count": len(env_to_vehicle),
            "env_to_vehicle_by_dataset": env_to_vehicle["dataset"].value_counts().to_dict(),
        }
    }
    
    out_json = ROOT / "outputs/e3_1_hybrid_pooling/data_audit_and_baseline_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nSaved audit summary to: {out_json}")
    
    print("\n=======================================================")
    print("Baseline vs E3.1 on Verified 8,213 Validation Partition")
    print("=======================================================")
    print(f"{'Metric':<25} | {'Baseline':<12} | {'E3.1':<12} | {'Gate Target':<12} | {'E3.1 Status'}")
    print("-" * 75)
    print(f"{'Macro F1':<25} | {b_rep['macro avg']['f1-score']:.4f}       | {e_rep['macro avg']['f1-score']:.4f}       | {'N/A':<12} | {'+' if e_rep['macro avg']['f1-score']>b_rep['macro avg']['f1-score'] else '-'}")
    print(f"{'Gunshot Recall':<25} | {b_rep['Gunshot_like']['recall']*100:.2f}%     | {e_rep['Gunshot_like']['recall']*100:.2f}%     | {'>= 94.0%':<12} | {'PASS' if e_gates['gunshot_recall_ge_94']['pass'] else 'FAIL'}")
    print(f"{'Human Precision':<25} | {b_rep['Human_Sound']['precision']*100:.2f}%     | {e_rep['Human_Sound']['precision']*100:.2f}%     | {'>= 40.0%':<12} | {'PASS' if e_gates['human_precision_ge_40']['pass'] else 'FAIL'}")
    print(f"{'Deforest Recall':<25} | {b_rep['Deforestation']['recall']*100:.2f}%     | {e_rep['Deforestation']['recall']*100:.2f}%     | {'>= 87.5%':<12} | {'PASS' if e_gates['deforest_recall_ge_87_5']['pass'] else 'FAIL'}")
    print(f"{'Vehicle Precision':<25} | {b_rep['Vehicle']['precision']*100:.2f}%     | {e_rep['Vehicle']['precision']*100:.2f}%     | {'>= 55.0%':<12} | {'PASS' if e_gates['vehicle_precision_ge_55']['pass'] else 'FAIL'}")

if __name__ == "__main__":
    main()
