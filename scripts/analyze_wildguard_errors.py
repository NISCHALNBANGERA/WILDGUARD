"""
WILDGUARD — Targeted Error Analysis & Data Audit
Traces Baseline and E2 misclassifications back to source recordings, datasets, and acoustic properties.
Audits training data for manual logging diversity, source imbalance, and label conflicts.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import AudioDataset, SmallCNN, CLASSES, LABEL_TO_ID

BASELINE_CKPT = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
E2_BEST_CKPT = ROOT / "outputs/audio_curated_augmented_e2/wildguard_audio_cnn_best.pt"
MANIFEST = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
OUT_DIR = ROOT / "outputs/wildguard_error_analysis"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest
    print(f"Loading manifest: {MANIFEST}")
    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)

    test_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "test")
    ].copy().reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")

    # 1. Run inference for Baseline and E2 to get per-sample logits and predictions
    loader = torch.utils.data.DataLoader(
        AudioDataset(test_df), batch_size=64, shuffle=False, num_workers=0
    )

    def run_model(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = SmallCNN(len(CLASSES)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        preds = []
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                max_prob, pred_idx = probs.max(dim=1)
                preds.extend(pred_idx.cpu().numpy().tolist())
                confs.extend(max_prob.cpu().numpy().tolist())
        return preds, confs

    print("Running Baseline inference...")
    bl_preds, bl_confs = run_model(BASELINE_CKPT)
    print("Running E2 Best inference...")
    e2_preds, e2_confs = run_model(E2_BEST_CKPT)

    # Attach predictions
    test_df["bl_pred_id"] = bl_preds
    test_df["bl_pred_class"] = test_df["bl_pred_id"].map(lambda i: CLASSES[i])
    test_df["bl_conf"] = bl_confs
    test_df["bl_correct"] = test_df["wildguard_class"] == test_df["bl_pred_class"]

    test_df["e2_pred_id"] = e2_preds
    test_df["e2_pred_class"] = test_df["e2_pred_id"].map(lambda i: CLASSES[i])
    test_df["e2_conf"] = e2_confs
    test_df["e2_correct"] = test_df["wildguard_class"] == test_df["e2_pred_class"]

    # 2. Extract Key Error Groups
    # Group 1: Gunshot_like -> Human_Sound
    gunshot_to_human_bl = test_df[(test_df["wildguard_class"] == "Gunshot_like") & (test_df["bl_pred_class"] == "Human_Sound")]
    gunshot_to_human_e2 = test_df[(test_df["wildguard_class"] == "Gunshot_like") & (test_df["e2_pred_class"] == "Human_Sound")]

    print(f"Gunshot_like -> Human_Sound errors: Baseline={len(gunshot_to_human_bl)}, E2={len(gunshot_to_human_e2)}")

    # Group 2: Deforestation -> Forest_Environment
    deforest_to_env_bl = test_df[(test_df["wildguard_class"] == "Deforestation") & (test_df["bl_pred_class"] == "Forest_Environment")]
    deforest_to_env_e2 = test_df[(test_df["wildguard_class"] == "Deforestation") & (test_df["e2_pred_class"] == "Forest_Environment")]

    print(f"Deforestation -> Forest_Environment errors: Baseline={len(deforest_to_env_bl)}, E2={len(deforest_to_env_e2)}")

    # Group 3: Forest_Environment -> Human_Sound
    env_to_human_bl = test_df[(test_df["wildguard_class"] == "Forest_Environment") & (test_df["bl_pred_class"] == "Human_Sound")]
    env_to_human_e2 = test_df[(test_df["wildguard_class"] == "Forest_Environment") & (test_df["e2_pred_class"] == "Human_Sound")]

    print(f"Forest_Environment -> Human_Sound errors: Baseline={len(env_to_human_bl)}, E2={len(env_to_human_e2)}")

    # Save detailed error CSVs
    err_cols = ["audio_path", "dataset", "source_label", "wildguard_class", "source_group_id",
                "bl_pred_class", "bl_conf", "bl_correct", "e2_pred_class", "e2_conf", "e2_correct"]
    
    test_df[err_cols].to_csv(OUT_DIR / "canonical_test_all_predictions.csv", index=False)
    
    # Save specific error subsets
    gunshot_to_human_e2[err_cols].to_csv(OUT_DIR / "gunshot_to_human_sound_e2_errors.csv", index=False)
    deforest_to_env_e2[err_cols].to_csv(OUT_DIR / "deforestation_to_environment_e2_errors.csv", index=False)
    env_to_human_e2[err_cols].to_csv(OUT_DIR / "environment_to_human_sound_e2_errors.csv", index=False)

    # 3. Analyze Gunshot_like -> Human_Sound Breakdown
    print("\n--- Gunshot_like -> Human_Sound Breakdown in E2 ---")
    print(gunshot_to_human_e2.groupby(["dataset", "source_label"]).size())

    # 4. Analyze Deforestation -> Forest_Environment Breakdown
    print("\n--- Deforestation -> Forest_Environment Breakdown in E2 ---")
    print(deforest_to_env_e2.groupby(["dataset", "source_label"]).size())

    # 5. Analyze Forest_Environment -> Human_Sound Breakdown
    print("\n--- Forest_Environment -> Human_Sound Breakdown in E2 ---")
    print(env_to_human_e2.groupby(["dataset", "source_label"]).size())

    # 6. Audit Training Data for Manual Logging
    print("\n--- Training Data Manual Logging Audit ---")
    train_df = df[
        (df["e2_eligible"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ].copy()

    manual_logging_subtypes = ["Axe", "WoodChop", "TreeFalling", "Handsaw", "hand_saw", "door_wood_knock", "door_wood_creaks"]
    manual_train = train_df[train_df["source_label"].isin(manual_logging_subtypes)]
    print(f"Total manual logging training samples: {len(manual_train)}")
    print(manual_train.groupby(["dataset", "source_label"])["source_group_id"].nunique())
    print("\nTotal Deforestation in training:", len(train_df[train_df["wildguard_class"] == "Deforestation"]))
    print(train_df[train_df["wildguard_class"] == "Deforestation"]["source_label"].value_counts())

    print("\nDone extracting initial error metrics.")

if __name__ == "__main__":
    main()
