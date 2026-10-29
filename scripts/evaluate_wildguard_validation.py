
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from train_wildguard_audio import (
    ROOT, MANIFEST, CLASSES, LABEL_TO_ID,
    AudioDataset, SmallCNN
)

OUT = ROOT / "outputs" / "validation_comparison"
EXPERIMENTS = {
    "pilot": ROOT / "outputs" / "audio_balanced_pilot" / "wildguard_audio_cnn_best.pt",
    "soft_balance": ROOT / "outputs" / "audio_soft_balance" / "wildguard_audio_cnn_best.pt",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)
    eligible = df["is_training_eligible_v3"].str.lower().eq("true")
    df = df.loc[eligible & df["wildguard_class"].isin(CLASSES)].copy()
    df["split_plan_v1"] = df["split_plan_v1"].str.lower().str.strip()

    val = df[df["split_plan_v1"] == "validation"].copy()
    if val.empty:
        raise RuntimeError("Validation split is empty.")

    loader = torch.utils.data.DataLoader(
        AudioDataset(val), batch_size=32, shuffle=False, num_workers=0
    )

    for name, checkpoint_path in EXPERIMENTS.items():
        if not checkpoint_path.is_file():
            print(f"SKIP {name}: checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if checkpoint.get("classes") != CLASSES:
            raise ValueError(f"{name}: checkpoint class mapping mismatch.")

        model = SmallCNN(len(CLASSES)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                y_true.extend(y.numpy().tolist())
                y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

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

        with open(OUT / f"{name}_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
            OUT / f"{name}_validation_confusion_matrix.csv"
        )

        print(f"\n{name}: accuracy={report['accuracy']:.4f}, "
              f"macro_f1={report['macro avg']['f1-score']:.4f}")
        for cls in CLASSES:
            print(f"  {cls}: precision={report[cls]['precision']:.3f}, "
                  f"recall={report[cls]['recall']:.3f}, "
                  f"F1={report[cls]['f1-score']:.3f}")

    print("\nValidation reports saved to:", OUT)


if __name__ == "__main__":
    main()
