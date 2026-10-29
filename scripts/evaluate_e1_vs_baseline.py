"""
WILDGUARD — Controlled Test-Set Evaluation: Baseline vs E1 Attempt 2
Evaluates both checkpoints on the identical held-out test split.
No training, no model selection, no checkpoint modification.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import AudioDataset, SmallCNN, CLASSES, LABEL_TO_ID

BASELINE_CKPT = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
E1_CKPT = ROOT / "outputs/audio_curated_subclass_balanced_v1/wildguard_audio_cnn_best.pt"
MANIFEST = ROOT / "data/processed/master_manifest_v14_e1_curated.csv"
OUT_DIR = ROOT / "outputs/e1_vs_baseline_test_evaluation"

def verify_checkpoints():
    """Verify both checkpoints are compatible."""
    bl = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=True)
    e1 = torch.load(E1_CKPT, map_location="cpu", weights_only=True)

    print("=== CHECKPOINT COMPATIBILITY ===")
    print(f"Baseline classes: {bl['classes']}")
    print(f"E1 classes:       {e1['classes']}")
    assert bl["classes"] == e1["classes"], "Class mapping mismatch!"

    for key in ["sample_rate", "duration_seconds", "n_mels", "n_fft", "hop_length"]:
        bv, ev = bl[key], e1[key]
        print(f"  {key}: baseline={bv}, e1={ev}, match={bv == ev}")
        assert bv == ev, f"Preprocessing mismatch on {key}!"

    bl_keys = sorted(bl["model_state_dict"].keys())
    e1_keys = sorted(e1["model_state_dict"].keys())
    assert bl_keys == e1_keys, "Model architecture mismatch!"
    for k in bl_keys:
        assert bl["model_state_dict"][k].shape == e1["model_state_dict"][k].shape, f"Shape mismatch: {k}"

    print(f"E1 best epoch: {e1.get('epoch', 'N/A')}")
    print("PASS: Both checkpoints are fully compatible.\n")
    return bl, e1


def verify_test_split():
    """Load test split, verify it matches baseline v13, confirm no leakage."""
    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    test = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "test")
    ].copy().reset_index(drop=True)

    # Cross-check against v13
    v13 = pd.read_csv(ROOT / "data/processed/master_manifest_v13_training_ready.csv", dtype=str, low_memory=False)
    v13_test = set(
        v13[(v13["is_training_eligible_v3"].str.lower() == "true") &
            (v13["split_plan_v1"].str.lower() == "test")]["audio_path"]
    )
    e1_test = set(test["audio_path"])
    assert v13_test == e1_test, f"Test set mismatch! v13={len(v13_test)}, e1={len(e1_test)}"

    # Verify no overlap with training
    train = df[
        (df["e1_eligible"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "train")
    ]
    train_groups = set(train["source_group_id"])
    test_groups = set(test["source_group_id"])
    overlap = train_groups & test_groups
    assert len(overlap) == 0, f"Leakage! {len(overlap)} groups in both train and test."

    print("=== TEST SPLIT VERIFICATION ===")
    print(f"Test samples: {len(test)}")
    print(f"Test identical to v13 baseline: True")
    print(f"Source group overlap with E1 training: 0")
    print(f"Test was NOT used for model selection (best checkpoint selected by val_loss).")
    print("PASS: Test set is untouched and leakage-safe.\n")

    print("Test class distribution:")
    print(test["wildguard_class"].value_counts().to_string())
    print()

    return test


def evaluate_checkpoint(ckpt_path, test_df, device, label):
    """Run forward pass and return metrics."""
    print(f"--- Evaluating: {label} ---")
    print(f"Checkpoint: {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SmallCNN(len(CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        AudioDataset(test_df), batch_size=64, shuffle=False, num_workers=0
    )

    all_y, all_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits = model(x.to(device))
            all_y.extend(y.numpy().tolist())
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            if (i + 1) % 40 == 0:
                print(f"  Processed {len(all_pred)}/{len(test_df)}...", flush=True)

    report_dict = classification_report(
        all_y, all_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        all_y, all_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    )
    cm = confusion_matrix(all_y, all_pred, labels=list(range(len(CLASSES))))

    print(report_text)
    print()

    return report_dict, cm, all_y, all_pred


def subtype_analysis(test_df, all_pred, label):
    """Per-subtype accuracy for key subtypes."""
    test_copy = test_df.copy()
    test_copy["pred_class"] = [CLASSES[i] for i in all_pred]

    print(f"--- Subtype Analysis: {label} ---")
    records = []
    for (cls, ds, sl), g in test_copy.groupby(["wildguard_class", "dataset", "source_label"]):
        total = len(g)
        correct = (g["pred_class"] == cls).sum()
        acc = correct / total * 100 if total > 0 else 0.0
        records.append({
            "class": cls, "dataset": ds, "subtype": sl,
            "samples": total, "correct": int(correct), "accuracy": round(acc, 1)
        })

    sub_df = pd.DataFrame(records)

    # Show key subtypes
    focus_classes = ["Deforestation", "Vehicle"]
    for fc in focus_classes:
        fc_df = sub_df[sub_df["class"] == fc].sort_values("accuracy")
        print(f"\n  {fc} subtypes:")
        for _, row in fc_df.iterrows():
            print(f"    {row['dataset']}/{row['subtype']}: {row['correct']}/{row['samples']} = {row['accuracy']}%")

    print()
    return sub_df


def main():
    # Step 1: Verify compatibility
    bl_ckpt, e1_ckpt = verify_checkpoints()

    # Step 2: Verify test split
    test_df = verify_test_split()

    # Step 3: Set up output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Step 4: Evaluate baseline
    bl_report, bl_cm, bl_y, bl_pred = evaluate_checkpoint(BASELINE_CKPT, test_df, device, "BASELINE")
    bl_subtypes = subtype_analysis(test_df, bl_pred, "BASELINE")

    # Step 5: Evaluate E1
    e1_report, e1_cm, e1_y, e1_pred = evaluate_checkpoint(E1_CKPT, test_df, device, "E1 ATTEMPT 2 (EPOCH 10)")
    e1_subtypes = subtype_analysis(test_df, e1_pred, "E1 ATTEMPT 2")

    # Step 6: Save results
    pd.DataFrame(bl_cm, index=CLASSES, columns=CLASSES).to_csv(OUT_DIR / "baseline_confusion_matrix.csv")
    pd.DataFrame(e1_cm, index=CLASSES, columns=CLASSES).to_csv(OUT_DIR / "e1_confusion_matrix.csv")
    with open(OUT_DIR / "baseline_test_report.json", "w") as f:
        json.dump(bl_report, f, indent=2)
    with open(OUT_DIR / "e1_test_report.json", "w") as f:
        json.dump(e1_report, f, indent=2)
    bl_subtypes.to_csv(OUT_DIR / "baseline_subtype_test.csv", index=False)
    e1_subtypes.to_csv(OUT_DIR / "e1_subtype_test.csv", index=False)

    # Step 7: Print comparison summary
    print("=" * 70)
    print("COMPARISON SUMMARY: BASELINE vs E1 ATTEMPT 2")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':>12} {'E1 (Ep10)':>12} {'Delta':>10}")
    print("-" * 70)

    metrics = [
        ("Accuracy", bl_report["accuracy"], e1_report["accuracy"]),
        ("Macro F1", bl_report["macro avg"]["f1-score"], e1_report["macro avg"]["f1-score"]),
        ("Weighted F1", bl_report["weighted avg"]["f1-score"], e1_report["weighted avg"]["f1-score"]),
    ]
    for name, bv, ev in metrics:
        delta = ev - bv
        sign = "+" if delta >= 0 else ""
        print(f"{name:<30} {bv:>11.4f} {ev:>12.4f} {sign}{delta:>9.4f}")

    print("-" * 70)
    print(f"{'Per-Class':^70}")
    print("-" * 70)
    print(f"{'Class':<22} {'BL Prec':>8} {'E1 Prec':>8} {'BL Rec':>7} {'E1 Rec':>7} {'BL F1':>6} {'E1 F1':>6}")
    print("-" * 70)
    for cls in CLASSES:
        bp = bl_report[cls]["precision"]
        ep = e1_report[cls]["precision"]
        br = bl_report[cls]["recall"]
        er = e1_report[cls]["recall"]
        bf = bl_report[cls]["f1-score"]
        ef = e1_report[cls]["f1-score"]
        print(f"{cls:<22} {bp:>8.4f} {ep:>8.4f} {br:>7.4f} {er:>7.4f} {bf:>6.4f} {ef:>6.4f}")

    print("=" * 70)
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
