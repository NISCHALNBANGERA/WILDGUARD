from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_wildguard_audio import (
    ROOT, MANIFEST, CLASSES, LABEL_TO_ID,
    AudioDataset, SmallCNN
)

OUT = ROOT / "outputs" / "validation_errors"
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

    # Reset index to guarantee alignment with AudioDataset.reset_index(drop=True)
    val = val.reset_index(drop=True)

    loader = torch.utils.data.DataLoader(
        AudioDataset(val), batch_size=32, shuffle=False, num_workers=0
    )

    for name, checkpoint_path in EXPERIMENTS.items():
        if not checkpoint_path.is_file():
            print(f"SKIP {name}: checkpoint not found: {checkpoint_path}")
            continue

        out_csv = OUT / f"{name}_misclassified_validation.csv"
        if out_csv.exists():
            raise FileExistsError(
                f"Output file already exists and overwrite protection is enabled: {out_csv}"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if checkpoint.get("classes") != CLASSES:
            raise ValueError(f"{name}: checkpoint class mapping mismatch.")

        model = SmallCNN(len(CLASSES)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        records = []
        val_idx = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(1)

                for i in range(len(y)):
                    true_label_id = y[i].item()
                    pred_label_id = preds[i].item()

                    if true_label_id != pred_label_id:
                        row_data = val.iloc[val_idx + i]
                        audio_path = row_data.get("audio_path", "")

                        true_class_name = CLASSES[true_label_id]
                        pred_class_name = CLASSES[pred_label_id]

                        pred_conf = probs[i, pred_label_id].item()
                        true_conf = probs[i, true_label_id].item()

                        records.append({
                            "audio_path": audio_path,
                            "true_class": true_class_name,
                            "predicted_class": pred_class_name,
                            "predicted_confidence": pred_conf,
                            "true_confidence": true_conf,
                        })
                val_idx += len(y)

        err_df = pd.DataFrame(records)
        err_df.to_csv(out_csv, index=False)
        print(f"[{name}] Exported {len(err_df)} misclassified validation samples to {out_csv}")

    print("\nAll validation error reports successfully generated in:", OUT)


if __name__ == "__main__":
    main()
