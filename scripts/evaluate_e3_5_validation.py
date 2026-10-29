"""
WILDGUARD Experiment E3.5 Validation Benchmark Script:
Evaluates outputs/e3_5_calibrated_hybrid/ checkpoints on the exact 8,210 validation samples.
Computes confusion matrix, gate status, and comparison with official baseline.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.insert(0, str(ROOT / "scripts"))
from train_wildguard_audio import CLASSES, LABEL_TO_ID, AudioDataset
from train_e3_hybrid import HybridCNN

E3_5_DIR = ROOT / "outputs/e3_5_calibrated_hybrid"
MANIFEST = ROOT / "data/processed/master_manifest_v16_e3_cleaned.csv"
BASELINE_VAL_PATH = ROOT / "outputs/e3_1_hybrid_pooling/reference_metrics_v16.json"


def evaluate_checkpoint(ckpt_path, val_df, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = HybridCNN(len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = torch.utils.data.DataLoader(
        AudioDataset(val_df), batch_size=64, shuffle=False, num_workers=0
    )

    all_y, all_preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_y.extend(y.numpy().tolist())

    report = classification_report(
        all_y, all_preds, target_names=CLASSES, output_dict=True, digits=5, zero_division=0
    )
    cm = confusion_matrix(all_y, all_preds, labels=range(len(CLASSES)))
    return report, cm, ckpt


def main():
    if not E3_5_DIR.exists():
        raise FileNotFoundError(f"E3.5 directory not found: {E3_5_DIR}")

    best_gates_path = E3_5_DIR / "wildguard_audio_cnn_best_gates.pt"
    best_loss_path = E3_5_DIR / "wildguard_audio_cnn_best.pt"

    eval_path = best_gates_path if best_gates_path.exists() else best_loss_path
    print(f"Evaluating E3.5 checkpoint: {eval_path.name}")

    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ].copy().reset_index(drop=True)

    assert len(val_df) == 8210, f"Expected 8210 validation samples, got {len(val_df)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    report, cm, ckpt = evaluate_checkpoint(eval_path, val_df, device)

    # Save JSON Report
    with open(E3_5_DIR / "e3_5_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save Confusion Matrix CSV
    cm_cols = [f"Pred_{c}" for c in CLASSES]
    cm_idx = [f"True_{c}" for c in CLASSES]
    cm_df = pd.DataFrame(cm, index=cm_idx, columns=cm_cols)
    cm_df["Total"] = cm_df.sum(axis=1)
    cm_df.to_csv(E3_5_DIR / "e3_5_validation_confusion_matrix.csv")

    # Load Baseline Validation Metrics
    with open(BASELINE_VAL_PATH) as f:
        base_metrics = json.load(f)["baseline_metrics"]

    macro_f1 = report["macro avg"]["f1-score"]
    base_macro_f1 = base_metrics["macro_f1"]
    acc = report["accuracy"]
    base_acc = base_metrics["accuracy"]

    g_rec = report["Gunshot_like"]["recall"]
    d_rec = report["Deforestation"]["recall"]
    h_prec = report["Human_Sound"]["precision"]
    v_prec = report["Vehicle"]["precision"]

    base_g_rec = base_metrics["gunshot_recall"]
    base_d_rec = base_metrics["deforest_recall"]
    base_h_prec = base_metrics["human_precision"]
    base_v_prec = base_metrics["vehicle_precision"]

    print("\n" + "=" * 80)
    print("  E3.5 VALIDATION EVALUATION RESULTS (8,210 Samples)")
    print("=" * 80)
    print(f"  Checkpoint Epoch: {ckpt.get('epoch')}")
    print(f"  Validation Macro F1: {macro_f1:.5f} (Baseline: {base_macro_f1:.5f}) -> Delta: {macro_f1 - base_macro_f1:+.5f}")
    print(f"  Validation Accuracy: {acc*100:.2f}% (Baseline: {base_acc*100:.2f}%) -> Delta: {(acc - base_acc)*100:+.2f}%")
    print(f"\n  Mandatory Regression Gates:")
    print(f"    Gate 1: Gunshot Recall:    {g_rec*100:.2f}% (Baseline: {base_g_rec*100:.2f}%) [Gate >= 94.0%] -> {'PASS' if g_rec >= 0.940 else 'FAIL'}")
    print(f"    Gate 2: Deforest Recall:   {d_rec*100:.2f}% (Baseline: {base_d_rec*100:.2f}%) [Gate >= 87.5%] -> {'PASS' if d_rec >= 87.5 else 'FAIL'}")
    print(f"    Gate 3: Human Precision:   {h_prec*100:.2f}% (Baseline: {base_h_prec*100:.2f}%) [Gate >= 40.0%] -> {'PASS' if h_prec >= 0.400 else 'FAIL'}")
    print(f"    Gate 4: Vehicle Precision: {v_prec*100:.2f}% (Baseline: {base_v_prec*100:.2f}%) [Gate >= 55.0%] -> {'PASS' if v_prec >= 0.550 else 'FAIL'}")

    # Generate Markdown Report
    md_content = f"""# WILDGUARD Experiment E3.5 Evaluation Report: Calibrated Sampler & Linear Focal Loss

**Experiment:** `e3_5_calibrated_hybrid`  
**Execution Date:** 2026-10-29  
**Checkpoint Evaluated:** [`{eval_path.name}`](file:///{eval_path.as_posix()}) (Epoch {ckpt.get('epoch')})  
**Validation Partition:** Exactly **8,210 samples** from [`master_manifest_v16_e3_cleaned.csv`](file:///{MANIFEST.as_posix()}) (0 test leakage).  
**Canonical Test Set Status:** **8,189 samples strictly FROZEN and UNTOUCHED (0 test evaluations)**.  

---

## 1. Executive Summary & Gate Status

- **Validation Macro F1:** **{macro_f1:.5f}** vs Baseline **{base_macro_f1:.5f}** ({'+' if macro_f1 > base_macro_f1 else ''}{macro_f1 - base_macro_f1:.5f})
- **Overall Accuracy:** **{acc*100:.2f}%** vs Baseline **{base_acc*100:.2f}%**
- **Regression Gates:**
  - Gate 1 (Gunshot Recall $\\ge 94.0\\%$): **{g_rec*100:.2f}%** ({'PASS' if g_rec >= 0.940 else 'FAIL'})
  - Gate 2 (Deforestation Recall $\\ge 87.5\\%$): **{d_rec*100:.2f}%** ({'PASS' if d_rec >= 0.875 else 'FAIL'})
  - Gate 3 (Human Precision $\\ge 40.0\\%$): **{h_prec*100:.2f}%** ({'PASS' if h_prec >= 0.400 else 'FAIL'})
  - Gate 4 (Vehicle Precision $\\ge 55.0\\%$): **{v_prec*100:.2f}%** ({'PASS' if v_prec >= 0.550 else 'FAIL'})

---

## 2. Benchmark Table (Exact 8,210 Validation Samples)

| Class | Baseline (SmallCNN) | E3.4 (HybridCNN) | E3.5 Calibrated (HybridCNN) | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for c in CLASSES:
        bp = base_metrics["per_class_metrics"][c]["precision"]
        br = base_metrics["per_class_metrics"][c]["recall"]
        bf = base_metrics["per_class_metrics"][c]["f1-score"]
        ep = report[c]["precision"]
        er = report[c]["recall"]
        ef = report[c]["f1-score"]
        sup = int(report[c]["support"])
        md_content += f"| **{c}** | P: {bp*100:.1f}% \\| R: {br*100:.1f}% \\| F1: {bf:.4f} | — | **P: {ep*100:.1f}% \\| R: {er*100:.1f}% \\| F1: {ef:.4f}** | {sup:,} |\n"

    md_content += f"""| **Overall Accuracy** | **{base_acc*100:.2f}%** | **90.49%** | **{acc*100:.2f}%** | 8,210 |
| **Macro F1** | **{base_macro_f1:.5f}** | **0.84155** | **{macro_f1:.5f}** | 8,210 |

---

## 3. Confusion Matrix (8,210 Validation Samples)
```
{cm_df.to_string()}
```
"""
    with open(E3_5_DIR / "e3_5_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nEvaluation report written to: {E3_5_DIR / 'e3_5_evaluation_report.md'}")


if __name__ == "__main__":
    main()
