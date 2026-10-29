from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from train_wildguard_audio import (
    ROOT, MANIFEST, CLASSES, LABEL_TO_ID,
    AudioDataset, SmallCNN
)

OUT = ROOT / "outputs" / "validation_hard_negative_comparison"
CHECKPOINT_PATH = ROOT / "outputs" / "audio_hard_negative_experiment_rerun" / "wildguard_audio_cnn_best.pt"
EXPERIMENT_NAME = "hard_negative"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Evaluating checkpoint:", CHECKPOINT_PATH)
    print("Output directory:", OUT)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    eligible = df["is_training_eligible_v3"].str.lower().eq("true")
    df = df.loc[eligible & df["wildguard_class"].isin(CLASSES)].copy()
    df["split_plan_v1"] = df["split_plan_v1"].str.lower().str.strip()

    val = df[df["split_plan_v1"] == "validation"].copy()
    if val.empty:
        raise RuntimeError("Validation split is empty.")

    print(f"Validation samples: {len(val)}")

    loader = torch.utils.data.DataLoader(
        AudioDataset(val), batch_size=32, shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=True
    )
    if checkpoint.get("classes") != CLASSES:
        raise ValueError(f"{EXPERIMENT_NAME}: checkpoint class mapping mismatch.")

    model = SmallCNN(len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    total_batches = len(loader)
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            logits = model(x.to(device))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - t0
                samples_done = len(y_true)
                print(f"Batch {batch_idx + 1}/{total_batches} ({samples_done}/{len(val)} samples, "
                      f"{elapsed:.1f}s, {samples_done/elapsed:.1f} samples/s)")

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )

    report_path = OUT / f"{EXPERIMENT_NAME}_validation_report.json"
    cm_path = OUT / f"{EXPERIMENT_NAME}_validation_confusion_matrix.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(cm_path)

    print(f"\n{EXPERIMENT_NAME}: accuracy={report['accuracy']:.4f}, "
          f"macro_f1={report['macro avg']['f1-score']:.4f}")
    for cls in CLASSES:
        print(f"  {cls}: precision={report[cls]['precision']:.4f}, "
              f"recall={report[cls]['recall']:.4f}, "
              f"F1={report[cls]['f1-score']:.4f}, "
              f"support={int(report[cls]['support'])}")

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    print("\nValidation reports saved to:", OUT)


if __name__ == "__main__":
    main()
