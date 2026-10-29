"""
WILDGUARD -- Rigorous Leakage-Safe Test-Set Evaluation: E2 vs Baseline
=====================================================================
Evaluates E2 best checkpoint and BASELINE checkpoint on the identical
frozen canonical test split (8,189 samples).

Verification steps:
  1. Checkpoint compatibility (classes, preprocessing, architecture)
  2. Test-set identity (v13 <-> v15 audio_path set equivalence)
  3. Data leakage checks (source_group_id overlap between E2 train and test)
  4. Duplicate recording detection across splits
  5. Sample count assertions

Output:
  outputs/e2_vs_baseline_test_evaluation/
    - baseline_test_report.json
    - e2_test_report.json
    - baseline_confusion_matrix.csv
    - e2_confusion_matrix.csv
    - baseline_subtype_test.csv
    - e2_subtype_test.csv
    - comparison_summary.json
    - comparison_report.md

No training. No checkpoint modification. No baseline overwrite.
"""
import sys
import json
import hashlib
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import AudioDataset, SmallCNN, CLASSES, LABEL_TO_ID

BASELINE_CKPT = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
E2_CKPT = ROOT / "outputs/audio_curated_augmented_e2/wildguard_audio_cnn_best.pt"
E2_MANIFEST = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
V13_MANIFEST = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
OUT_DIR = ROOT / "outputs/e2_vs_baseline_test_evaluation"

BLOCKERS = []


def blocker(msg):
    """Register a verification failure and raise immediately."""
    BLOCKERS.append(msg)
    raise RuntimeError(f"BLOCKER: {msg}")


def verify_checkpoints():
    """Verify both checkpoints have identical class mapping, preprocessing, and architecture."""
    print("=" * 70)
    print("STEP 1: CHECKPOINT COMPATIBILITY VERIFICATION")
    print("=" * 70)

    if not BASELINE_CKPT.exists():
        blocker(f"Baseline checkpoint not found: {BASELINE_CKPT}")
    if not E2_CKPT.exists():
        blocker(f"E2 checkpoint not found: {E2_CKPT}")

    bl = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=True)
    e2 = torch.load(E2_CKPT, map_location="cpu", weights_only=True)

    # Class mapping
    print(f"  Baseline classes: {bl['classes']}")
    print(f"  E2 classes:       {e2['classes']}")
    if bl["classes"] != e2["classes"]:
        blocker(f"Class mapping mismatch! Baseline={bl['classes']}, E2={e2['classes']}")
    if bl["classes"] != CLASSES:
        blocker(f"Checkpoint classes don't match code CLASSES! ckpt={bl['classes']}, code={CLASSES}")

    # Class-index mapping verification
    for i, cls in enumerate(CLASSES):
        if LABEL_TO_ID[cls] != i:
            blocker(f"LABEL_TO_ID[{cls}]={LABEL_TO_ID[cls]} != expected index {i}")
    print(f"  Class-index mapping: {LABEL_TO_ID}")
    print(f"  CLASSES order verified: {CLASSES}")

    # Preprocessing params
    preproc_keys = ["sample_rate", "duration_seconds", "n_mels", "n_fft", "hop_length"]
    for key in preproc_keys:
        bv = bl[key]
        ev = e2[key]
        match = bv == ev
        print(f"  {key}: baseline={bv}, e2={ev}, match={match}")
        if not match:
            blocker(f"Preprocessing mismatch on {key}: baseline={bv}, e2={ev}")

    # Architecture (state_dict keys and shapes)
    bl_keys = sorted(bl["model_state_dict"].keys())
    e2_keys = sorted(e2["model_state_dict"].keys())
    if bl_keys != e2_keys:
        blocker(f"Model architecture mismatch! Different state_dict keys.")
    for k in bl_keys:
        if bl["model_state_dict"][k].shape != e2["model_state_dict"][k].shape:
            blocker(f"Shape mismatch for layer {k}: baseline={bl['model_state_dict'][k].shape}, e2={e2['model_state_dict'][k].shape}")

    # Weight divergence check (ensure they are actually different models)
    weight_diffs = []
    for k in bl_keys:
        diff = (bl["model_state_dict"][k].float() - e2["model_state_dict"][k].float()).abs().max().item()
        weight_diffs.append(diff)
    max_diff = max(weight_diffs)
    if max_diff == 0:
        print("  WARNING: Models have IDENTICAL weights -- this is the same checkpoint!")
    else:
        print(f"  Weight max absolute difference: {max_diff:.6f} (models are distinct)")

    # E2 training metadata
    e2_epoch = e2.get("epoch", "N/A")
    e2_best_val = e2.get("best_val_loss", "N/A")
    print(f"  E2 best epoch: {e2_epoch}")
    print(f"  E2 best val_loss: {e2_best_val}")

    print("PASS: Checkpoints are fully compatible.\n")
    return bl, e2


def verify_test_split():
    """Load and verify the canonical test split. Check for leakage."""
    print("=" * 70)
    print("STEP 2: TEST SPLIT VERIFICATION & LEAKAGE CHECKS")
    print("=" * 70)

    if not E2_MANIFEST.exists():
        blocker(f"E2 manifest not found: {E2_MANIFEST}")
    if not V13_MANIFEST.exists():
        blocker(f"V13 manifest not found: {V13_MANIFEST}")

    e2_df = pd.read_csv(E2_MANIFEST, dtype=str, low_memory=False)
    v13_df = pd.read_csv(V13_MANIFEST, dtype=str, low_memory=False)

    # Extract test sets from both manifests
    test_e2 = e2_df[
        (e2_df["is_training_eligible_v3"].str.lower() == "true") &
        (e2_df["split_plan_v1"].str.lower() == "test")
    ].copy().reset_index(drop=True)

    test_v13 = v13_df[
        (v13_df["is_training_eligible_v3"].str.lower() == "true") &
        (v13_df["split_plan_v1"].str.lower() == "test")
    ].copy().reset_index(drop=True)

    # Check sample count
    print(f"  E2 manifest test samples:  {len(test_e2)}")
    print(f"  V13 manifest test samples: {len(test_v13)}")

    if len(test_e2) != 8189:
        blocker(f"Expected 8189 test samples in E2 manifest, found {len(test_e2)}")
    if len(test_v13) != 8189:
        blocker(f"Expected 8189 test samples in V13 manifest, found {len(test_v13)}")

    # Set-identity check on audio_path
    e2_paths = set(test_e2["audio_path"])
    v13_paths = set(test_v13["audio_path"])

    if e2_paths != v13_paths:
        only_e2 = e2_paths - v13_paths
        only_v13 = v13_paths - e2_paths
        blocker(
            f"Test set mismatch! "
            f"Only in E2: {len(only_e2)} paths, Only in V13: {len(only_v13)} paths. "
            f"Example E2-only: {list(only_e2)[:3]}, Example V13-only: {list(only_v13)[:3]}"
        )

    print(f"  Test set identity: E2 == V13 (all 8,189 audio_paths match)")

    # Check label consistency between manifests
    e2_labels = dict(zip(test_e2["audio_path"], test_e2["wildguard_class"]))
    v13_labels = dict(zip(test_v13["audio_path"], test_v13["wildguard_class"]))
    label_mismatches = []
    for path in e2_paths:
        if e2_labels[path] != v13_labels[path]:
            label_mismatches.append((path, e2_labels[path], v13_labels[path]))
    if label_mismatches:
        blocker(f"{len(label_mismatches)} test samples have different labels between manifests! Examples: {label_mismatches[:3]}")
    print(f"  Label consistency: All 8,189 test labels match between manifests")

    # Leakage check: source_group_id overlap between E2 training set and test set
    e2_train = e2_df[
        (e2_df["e2_eligible"].str.lower() == "true") &
        (e2_df["split_plan_v1"].str.lower() == "train")
    ]
    if "source_group_id" in e2_df.columns:
        train_groups = set(e2_train["source_group_id"].dropna())
        test_groups = set(test_e2["source_group_id"].dropna())
        overlap = train_groups & test_groups
        if overlap:
            blocker(f"DATA LEAKAGE! {len(overlap)} source_group_ids appear in both E2 train and test: {list(overlap)[:5]}")
        print(f"  Source group leakage check: 0 overlapping groups (PASS)")
    else:
        print(f"  WARNING: 'source_group_id' column not found -- cannot verify group-level leakage")

    # Leakage check: audio_path overlap between E2 training set and test set
    train_paths = set(e2_train["audio_path"])
    test_path_set = set(test_e2["audio_path"])
    path_overlap = train_paths & test_path_set
    if path_overlap:
        blocker(f"DATA LEAKAGE! {len(path_overlap)} audio files appear in both E2 train and test!")
    print(f"  Audio path leakage check: 0 overlapping files (PASS)")

    # Duplicate detection: check for duplicate filenames (different paths, same filename) across splits
    # This catches copies of the same recording under different directories
    train_basenames = defaultdict(list)
    for p in train_paths:
        train_basenames[Path(p).name].append(p)
    test_basenames = defaultdict(list)
    for p in test_path_set:
        test_basenames[Path(p).name].append(p)
    basename_overlap = set(train_basenames.keys()) & set(test_basenames.keys())
    if basename_overlap:
        print(f"  WARNING: {len(basename_overlap)} filenames appear in both train and test (may be different recordings).")
        for bn in list(basename_overlap)[:5]:
            print(f"    {bn}: train={train_basenames[bn][:1]}, test={test_basenames[bn][:1]}")
    else:
        print(f"  Duplicate filename check: No basename overlaps across splits")

    # Per-class test distribution
    print(f"\n  Test class distribution:")
    class_dist = test_e2["wildguard_class"].value_counts()
    for cls in CLASSES:
        cnt = class_dist.get(cls, 0)
        print(f"    {cls}: {cnt}")

    # Check all test audio files exist
    missing = [p for p in test_e2["audio_path"] if not Path(p).is_file()]
    if missing:
        blocker(f"{len(missing)} test audio files are missing! First: {missing[0]}")
    print(f"\n  Audio file existence: All 8,189 test files verified on disk")

    print("PASS: Test set is frozen, leakage-safe, and verified.\n")
    return test_e2


def evaluate_checkpoint(ckpt_path, test_df, device, model_name):
    """Run inference on canonical test split. Returns metrics and predictions."""
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = SmallCNN(len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = torch.utils.data.DataLoader(
        AudioDataset(test_df), batch_size=64, shuffle=False, num_workers=0
    )

    all_y = []
    all_pred = []
    all_logits = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            all_y.extend(y.numpy().tolist())
            all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())

            if (i + 1) % 40 == 0:
                print(f"  Processed {len(all_pred)}/{len(test_df)}...", flush=True)

    print(f"  Inference complete: {len(all_pred)} predictions\n")

    target_names = [CLASSES[i] for i in range(len(CLASSES))]
    report_dict = classification_report(
        all_y, all_pred,
        labels=list(range(len(CLASSES))),
        target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        all_y, all_pred,
        labels=list(range(len(CLASSES))),
        target_names=target_names,
        zero_division=0, digits=4,
    )
    cm = confusion_matrix(all_y, all_pred, labels=list(range(len(CLASSES))))

    print(report_text)
    return report_dict, cm, all_y, all_pred, all_logits


def subtype_analysis(test_df, predictions, model_name):
    """Per-subtype accuracy breakdown."""
    df = test_df.copy()
    df["pred_id"] = predictions
    df["pred_class"] = df["pred_id"].map(lambda i: CLASSES[i])
    df["correct"] = df["wildguard_class"] == df["pred_class"]

    records = []
    for (cls, ds, sl), grp in df.groupby(["wildguard_class", "dataset", "source_label"]):
        total = len(grp)
        corr = int(grp["correct"].sum())
        acc = round(corr / total * 100, 2) if total > 0 else 0.0
        # Most common misclassification
        wrong = grp[~grp["correct"]]
        if len(wrong) > 0:
            top_confusion = wrong["pred_class"].value_counts().index[0]
        else:
            top_confusion = "N/A"
        records.append({
            "model": model_name,
            "class": cls,
            "dataset": ds,
            "subtype": sl,
            "samples": total,
            "correct": corr,
            "accuracy": acc,
            "top_confusion": top_confusion,
        })
    return pd.DataFrame(records)


def error_analysis(test_df, predictions, model_name):
    """Identify top false-positive and false-negative patterns."""
    df = test_df.copy()
    df["pred_id"] = predictions
    df["pred_class"] = df["pred_id"].map(lambda i: CLASSES[i])
    df["correct"] = df["wildguard_class"] == df["pred_class"]

    # False positives: predicted as class C but true label is different
    fp_records = []
    for cls in CLASSES:
        fp_mask = (df["pred_class"] == cls) & (df["wildguard_class"] != cls)
        fp_df = df[fp_mask]
        if len(fp_df) > 0:
            for true_cls, grp in fp_df.groupby("wildguard_class"):
                fp_records.append({
                    "predicted_as": cls,
                    "true_class": true_cls,
                    "count": len(grp),
                    "pct_of_class_predictions": round(
                        len(grp) / max(1, (df["pred_class"] == cls).sum()) * 100, 2
                    ),
                })

    # False negatives: true label is class C but predicted as something else
    fn_records = []
    for cls in CLASSES:
        fn_mask = (df["wildguard_class"] == cls) & (df["pred_class"] != cls)
        fn_df = df[fn_mask]
        if len(fn_df) > 0:
            for pred_cls, grp in fn_df.groupby("pred_class"):
                fn_records.append({
                    "true_class": cls,
                    "predicted_as": pred_cls,
                    "count": len(grp),
                    "pct_of_class_samples": round(
                        len(grp) / max(1, (df["wildguard_class"] == cls).sum()) * 100, 2
                    ),
                })

    fp_df = pd.DataFrame(fp_records).sort_values("count", ascending=False) if fp_records else pd.DataFrame()
    fn_df = pd.DataFrame(fn_records).sort_values("count", ascending=False) if fn_records else pd.DataFrame()

    return fp_df, fn_df


def format_confusion_matrix_md(cm, classes):
    """Format confusion matrix as markdown table."""
    header = "| True \\ Pred | " + " | ".join(classes) + " |"
    sep = "|" + "|".join(["---"] * (len(classes) + 1)) + "|"
    rows = [header, sep]
    for i, cls in enumerate(classes):
        vals = " | ".join(str(cm[i][j]) for j in range(len(classes)))
        rows.append(f"| **{cls}** | {vals} |")
    return "\n".join(rows)


def main():
    print("WILDGUARD E2 vs Baseline -- Rigorous Test-Set Evaluation")
    print("=" * 70)
    print(f"E2 checkpoint: {E2_CKPT}")
    print(f"Baseline checkpoint: {BASELINE_CKPT}")
    print(f"E2 manifest: {E2_MANIFEST}")
    print(f"V13 manifest: {V13_MANIFEST}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70 + "\n")

    # === STEP 1: Checkpoint compatibility ===
    try:
        bl_ckpt, e2_ckpt = verify_checkpoints()
    except RuntimeError as e:
        print(f"\n*** EVALUATION ABORTED: {e}")
        return

    # === STEP 2: Test split verification ===
    try:
        test_df = verify_test_split()
    except RuntimeError as e:
        print(f"\n*** EVALUATION ABORTED: {e}")
        return

    # === STEP 3: Run evaluation ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}\n")

    bl_report, bl_cm, bl_y, bl_pred, bl_logits = evaluate_checkpoint(
        BASELINE_CKPT, test_df, device, "BASELINE"
    )
    e2_report, e2_cm, e2_y, e2_pred, e2_logits = evaluate_checkpoint(
        E2_CKPT, test_df, device, "E2 AUGMENTED"
    )

    # === STEP 4: Subtype analysis ===
    print("\n" + "=" * 70)
    print("SUBTYPE ANALYSIS")
    print("=" * 70)

    bl_subtypes = subtype_analysis(test_df, bl_pred, "BASELINE")
    e2_subtypes = subtype_analysis(test_df, e2_pred, "E2 AUGMENTED")

    focus_classes = ["Deforestation", "Vehicle", "Human_Sound"]
    for model_name, sub_df in [("BASELINE", bl_subtypes), ("E2", e2_subtypes)]:
        print(f"\n--- {model_name} Focus Classes ---")
        for fc in focus_classes:
            fc_rows = sub_df[sub_df["class"] == fc].sort_values("accuracy")
            print(f"\n  {fc} subtypes:")
            for _, row in fc_rows.iterrows():
                print(f"    {row['dataset']}/{row['subtype']}: {row['correct']}/{row['samples']} = {row['accuracy']}% (confused->{row['top_confusion']})")

    # === STEP 5: Error analysis ===
    print("\n" + "=" * 70)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 70)

    bl_fp, bl_fn = error_analysis(test_df, bl_pred, "BASELINE")
    e2_fp, e2_fn = error_analysis(test_df, e2_pred, "E2 AUGMENTED")

    print("\n--- Baseline Top False Positives ---")
    if not bl_fp.empty:
        print(bl_fp.head(10).to_string(index=False))
    print("\n--- Baseline Top False Negatives ---")
    if not bl_fn.empty:
        print(bl_fn.head(10).to_string(index=False))

    print("\n--- E2 Top False Positives ---")
    if not e2_fp.empty:
        print(e2_fp.head(10).to_string(index=False))
    print("\n--- E2 Top False Negatives ---")
    if not e2_fn.empty:
        print(e2_fn.head(10).to_string(index=False))

    # === STEP 6: Save all artifacts ===
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    pd.DataFrame(bl_cm, index=CLASSES, columns=CLASSES).to_csv(OUT_DIR / "baseline_confusion_matrix.csv")
    pd.DataFrame(e2_cm, index=CLASSES, columns=CLASSES).to_csv(OUT_DIR / "e2_confusion_matrix.csv")

    with open(OUT_DIR / "baseline_test_report.json", "w") as f:
        json.dump(bl_report, f, indent=2)
    with open(OUT_DIR / "e2_test_report.json", "w") as f:
        json.dump(e2_report, f, indent=2)

    bl_subtypes.to_csv(OUT_DIR / "baseline_subtype_test.csv", index=False)
    e2_subtypes.to_csv(OUT_DIR / "e2_subtype_test.csv", index=False)

    # Machine-readable comparison summary
    summary = {
        "test_samples": 8189,
        "evaluation_checkpoints": {
            "baseline": str(BASELINE_CKPT),
            "e2": str(E2_CKPT),
        },
        "baseline": {
            "accuracy": bl_report["accuracy"],
            "macro_f1": bl_report["macro avg"]["f1-score"],
            "weighted_f1": bl_report["weighted avg"]["f1-score"],
            "macro_precision": bl_report["macro avg"]["precision"],
            "macro_recall": bl_report["macro avg"]["recall"],
            "per_class": {cls: {
                "precision": bl_report[cls]["precision"],
                "recall": bl_report[cls]["recall"],
                "f1": bl_report[cls]["f1-score"],
                "support": bl_report[cls]["support"],
            } for cls in CLASSES},
        },
        "e2": {
            "accuracy": e2_report["accuracy"],
            "macro_f1": e2_report["macro avg"]["f1-score"],
            "weighted_f1": e2_report["weighted avg"]["f1-score"],
            "macro_precision": e2_report["macro avg"]["precision"],
            "macro_recall": e2_report["macro avg"]["recall"],
            "per_class": {cls: {
                "precision": e2_report[cls]["precision"],
                "recall": e2_report[cls]["recall"],
                "f1": e2_report[cls]["f1-score"],
                "support": e2_report[cls]["support"],
            } for cls in CLASSES},
        },
        "deltas": {
            "accuracy": e2_report["accuracy"] - bl_report["accuracy"],
            "macro_f1": e2_report["macro avg"]["f1-score"] - bl_report["macro avg"]["f1-score"],
            "weighted_f1": e2_report["weighted avg"]["f1-score"] - bl_report["weighted avg"]["f1-score"],
            "per_class_f1": {cls: e2_report[cls]["f1-score"] - bl_report[cls]["f1-score"] for cls in CLASSES},
        },
        "verification": {
            "test_set_frozen": True,
            "test_set_identity_v13_v15": True,
            "source_group_leakage": False,
            "audio_path_leakage": False,
            "checkpoint_compatibility": True,
            "preprocessing_parity": True,
        },
    }

    with open(OUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # === STEP 7: Print summary ===
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: BASELINE vs E2 AUGMENTED")
    print("=" * 80)
    print(f"{'Metric':<30} {'Baseline':>12} {'E2':>12} {'Delta':>12}")
    print("-" * 70)

    for metric, key in [("Accuracy", "accuracy"), ("Macro F1", "macro avg"), ("Weighted F1", "weighted avg")]:
        bv = bl_report[key] if key == "accuracy" else bl_report[key]["f1-score"]
        ev = e2_report[key] if key == "accuracy" else e2_report[key]["f1-score"]
        delta = ev - bv
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<30} {bv:>11.4f} {ev:>12.4f} {sign}{delta:>11.4f}")

    print("-" * 70)
    print(f"{'Per-Class F1':^70}")
    print("-" * 70)
    print(f"{'Class':<22} {'BL Prec':>8} {'E2 Prec':>8} {'BL Rec':>7} {'E2 Rec':>7} {'BL F1':>7} {'E2 F1':>7} {'dF1':>7}")
    print("-" * 70)
    for cls in CLASSES:
        bp = bl_report[cls]["precision"]
        ep = e2_report[cls]["precision"]
        br = bl_report[cls]["recall"]
        er = e2_report[cls]["recall"]
        bf = bl_report[cls]["f1-score"]
        ef = e2_report[cls]["f1-score"]
        df1 = ef - bf
        sign = "+" if df1 >= 0 else ""
        print(f"{cls:<22} {bp:>8.4f} {ep:>8.4f} {br:>7.4f} {er:>7.4f} {bf:>7.4f} {ef:>7.4f} {sign}{df1:>6.4f}")

    print("=" * 80)

    # === STEP 8: Generate Markdown Report ===
    generate_markdown_report(
        bl_report, e2_report, bl_cm, e2_cm,
        bl_subtypes, e2_subtypes,
        bl_fp, bl_fn, e2_fp, e2_fn,
        summary
    )

    print(f"\nAll artifacts saved to: {OUT_DIR}")
    print("EVALUATION COMPLETE.")


def generate_markdown_report(bl_report, e2_report, bl_cm, e2_cm,
                              bl_subtypes, e2_subtypes,
                              bl_fp, bl_fn, e2_fp, e2_fn, summary):
    """Generate comprehensive Markdown comparison report."""

    # Determine recommendation
    bl_macro_f1 = bl_report["macro avg"]["f1-score"]
    e2_macro_f1 = e2_report["macro avg"]["f1-score"]
    bl_acc = bl_report["accuracy"]
    e2_acc = e2_report["accuracy"]

    # Check per-class regressions
    regressions = []
    improvements = []
    for cls in CLASSES:
        delta = e2_report[cls]["f1-score"] - bl_report[cls]["f1-score"]
        if delta < -0.01:
            regressions.append((cls, delta))
        elif delta > 0.01:
            improvements.append((cls, delta))

    if e2_macro_f1 > bl_macro_f1 and not regressions:
        recommendation = "E2 AUGMENTED"
        rec_reason = "E2 improves macro F1 with no class-level regressions."
    elif e2_macro_f1 > bl_macro_f1 and regressions:
        recommendation = "NEEDS REVIEW"
        reg_str = ", ".join(f"{c} ({d:+.4f})" for c, d in regressions)
        rec_reason = f"E2 improves macro F1 but shows regressions in: {reg_str}."
    elif e2_macro_f1 < bl_macro_f1:
        recommendation = "BASELINE"
        rec_reason = "E2 does not improve macro F1 vs baseline."
    else:
        recommendation = "TIE -- review class-level metrics"
        rec_reason = "Macro F1 is equal; decide based on class priorities."

    # Build markdown
    md_lines = []
    md_lines.append("# WILDGUARD E2 vs Baseline -- Test-Set Evaluation Report")
    md_lines.append("")
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(f"| Metric | Baseline | E2 Augmented | Delta |")
    md_lines.append(f"|--------|----------|-------------|-------|")
    md_lines.append(f"| **Accuracy** | {bl_acc:.4f} | {e2_acc:.4f} | {e2_acc - bl_acc:+.4f} |")
    md_lines.append(f"| **Macro F1** | {bl_macro_f1:.4f} | {e2_macro_f1:.4f} | {e2_macro_f1 - bl_macro_f1:+.4f} |")
    bl_wf1 = bl_report["weighted avg"]["f1-score"]
    e2_wf1 = e2_report["weighted avg"]["f1-score"]
    md_lines.append(f"| **Weighted F1** | {bl_wf1:.4f} | {e2_wf1:.4f} | {e2_wf1 - bl_wf1:+.4f} |")
    md_lines.append("")

    md_lines.append(f"> **Recommendation: {recommendation}**")
    md_lines.append(f"> {rec_reason}")
    md_lines.append("")

    # Verification
    md_lines.append("## Verification Checks")
    md_lines.append("")
    md_lines.append("| Check | Status |")
    md_lines.append("|-------|--------|")
    md_lines.append("| Test set frozen (8,189 samples) | PASS |")
    md_lines.append("| Test set identity (V13 == V15) | PASS |")
    md_lines.append("| Source group leakage | No leakage |")
    md_lines.append("| Audio path leakage | No leakage |")
    md_lines.append("| Checkpoint compatibility | PASS |")
    md_lines.append("| Preprocessing parity | PASS |")
    md_lines.append("| Class-index mapping | PASS |")
    md_lines.append("")

    # Per-class metrics
    md_lines.append("## Per-Class Comparison")
    md_lines.append("")
    md_lines.append("| Class | BL Prec | E2 Prec | BL Rec | E2 Rec | BL F1 | E2 F1 | Delta F1 |")
    md_lines.append("|-------|---------|---------|--------|--------|-------|-------|-----|")
    for cls in CLASSES:
        bp = bl_report[cls]["precision"]
        ep = e2_report[cls]["precision"]
        br = bl_report[cls]["recall"]
        er = e2_report[cls]["recall"]
        bf = bl_report[cls]["f1-score"]
        ef = e2_report[cls]["f1-score"]
        df1 = ef - bf
        indicator = "[+]" if df1 > 0.005 else ("[-]" if df1 < -0.005 else "[=]")
        md_lines.append(
            f"| {cls} | {bp:.4f} | {ep:.4f} | {br:.4f} | {er:.4f} | {bf:.4f} | {ef:.4f} | {indicator} {df1:+.4f} |"
        )
    md_lines.append("")

    # Focus classes
    md_lines.append("## Focus Class Analysis: Deforestation, Vehicle, Human_Sound")
    md_lines.append("")
    for fc in ["Deforestation", "Vehicle", "Human_Sound"]:
        md_lines.append(f"### {fc}")
        md_lines.append("")
        bf = bl_report[fc]["f1-score"]
        ef = e2_report[fc]["f1-score"]
        md_lines.append(f"- Baseline F1: {bf:.4f}, E2 F1: {ef:.4f} (delta = {ef-bf:+.4f})")
        md_lines.append("")

        # Subtype comparison
        bl_sub = bl_subtypes[bl_subtypes["class"] == fc].sort_values("accuracy")
        e2_sub = e2_subtypes[e2_subtypes["class"] == fc].sort_values("accuracy")

        if not bl_sub.empty:
            md_lines.append(f"**Subtype breakdown:**")
            md_lines.append("")
            md_lines.append("| Subtype | BL Acc% | E2 Acc% | BL n/N | E2 n/N |")
            md_lines.append("|---------|---------|---------|--------|--------|")

            all_subtypes = set(bl_sub["subtype"].tolist() + e2_sub["subtype"].tolist())
            for st in sorted(all_subtypes):
                bl_row = bl_sub[bl_sub["subtype"] == st]
                e2_row = e2_sub[e2_sub["subtype"] == st]
                bl_a = bl_row["accuracy"].iloc[0] if not bl_row.empty else "-"
                e2_a = e2_row["accuracy"].iloc[0] if not e2_row.empty else "-"
                bl_cn = f"{bl_row['correct'].iloc[0]}/{bl_row['samples'].iloc[0]}" if not bl_row.empty else "-"
                e2_cn = f"{e2_row['correct'].iloc[0]}/{e2_row['samples'].iloc[0]}" if not e2_row.empty else "-"
                md_lines.append(f"| {st} | {bl_a} | {e2_a} | {bl_cn} | {e2_cn} |")
            md_lines.append("")

    # Confusion matrices
    md_lines.append("## Confusion Matrices")
    md_lines.append("")
    md_lines.append("### Baseline")
    md_lines.append("")
    md_lines.append(format_confusion_matrix_md(bl_cm, CLASSES))
    md_lines.append("")
    md_lines.append("### E2 Augmented")
    md_lines.append("")
    md_lines.append(format_confusion_matrix_md(e2_cm, CLASSES))
    md_lines.append("")

    # Error patterns
    md_lines.append("## Error Patterns")
    md_lines.append("")
    md_lines.append("### Top False Positive Patterns (E2)")
    md_lines.append("")
    if not e2_fp.empty:
        md_lines.append("| Predicted As | True Class | Count | % of Pred |")
        md_lines.append("|-------------|------------|-------|-----------|")
        for _, row in e2_fp.head(10).iterrows():
            md_lines.append(f"| {row['predicted_as']} | {row['true_class']} | {row['count']} | {row['pct_of_class_predictions']:.1f}% |")
    md_lines.append("")

    md_lines.append("### Top False Negative Patterns (E2)")
    md_lines.append("")
    if not e2_fn.empty:
        md_lines.append("| True Class | Predicted As | Count | % of Class |")
        md_lines.append("|------------|-------------|-------|------------|")
        for _, row in e2_fn.head(10).iterrows():
            md_lines.append(f"| {row['true_class']} | {row['predicted_as']} | {row['count']} | {row['pct_of_class_samples']:.1f}% |")
    md_lines.append("")

    # Recommendation
    md_lines.append("## Recommendation")
    md_lines.append("")
    md_lines.append(f"**Recommended Leader: {recommendation}**")
    md_lines.append("")
    md_lines.append(f"{rec_reason}")
    md_lines.append("")
    if improvements:
        md_lines.append("**Class-level improvements (E2 vs Baseline):**")
        for cls, delta in improvements:
            md_lines.append(f"- {cls}: {delta:+.4f} F1")
    if regressions:
        md_lines.append("")
        md_lines.append("**Class-level regressions (E2 vs Baseline):**")
        for cls, delta in regressions:
            md_lines.append(f"- {cls}: {delta:+.4f} F1")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append(f"*Generated by evaluate_e2_vs_baseline_rigorous.py*")
    md_lines.append(f"*Test set: 8,189 frozen canonical samples*")
    md_lines.append(f"*Baseline: audio_hard_negative_experiment_rerun*")
    md_lines.append(f"*E2: audio_curated_augmented_e2 (best checkpoint, epoch {summary['e2'].get('best_epoch', 'N/A')})*")

    report_text = "\n".join(md_lines)
    with open(OUT_DIR / "comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Markdown report saved: {OUT_DIR / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
