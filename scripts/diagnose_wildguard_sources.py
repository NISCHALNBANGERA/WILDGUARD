
from pathlib import Path
import json

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
CHECKPOINT = ROOT / "outputs/audio_baseline/wildguard_audio_cnn_best.pt"
OUT = ROOT / "outputs/audio_diagnostics"

CLASSES = [
    "Deforestation",
    "Forest_Environment",
    "Forest_Fire",
    "Gunshot_like",
    "Human_Sound",
    "Vehicle",
]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SR, DURATION, N_MELS = 16000, 4, 64
N_SAMPLES = SR * DURATION
N_FFT, HOP = 400, 160


def load_audio(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    return audio


def make_features(path):
    audio = load_audio(path)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=40, fmax=SR // 2,
        center=False,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80, 0) / 80.0
    return torch.from_numpy(logmel.astype(np.float32)).unsqueeze(0)


class AudioDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return make_features(row.audio_path), LABEL_TO_ID[row.wildguard_class]


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    if not MANIFEST.exists() or not CHECKPOINT.exists():
        raise FileNotFoundError("Manifest or checkpoint not found.")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, low_memory=False)
    eligible = df["is_training_eligible_v3"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    test = df[
        eligible
        & df["split_plan_v1"].astype(str).str.lower().eq("test")
        & df["wildguard_class"].isin(CLASSES)
    ].copy()

    if test.empty:
        raise RuntimeError("No eligible test rows found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = SmallCNN(len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(AudioDataset(test), batch_size=16, shuffle=False, num_workers=0)
    predictions, actuals, confidence = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            predictions.extend(pred.cpu().tolist())
            actuals.extend(y.tolist())
            confidence.extend(conf.cpu().tolist())

    test["actual"] = [CLASSES[i] for i in actuals]
    test["predicted"] = [CLASSES[i] for i in predictions]
    test["confidence_percent"] = np.array(confidence) * 100
    test["correct"] = test["actual"].eq(test["predicted"])

    # Full rows preserve source and split metadata for auditing.
    test.to_csv(OUT / "source_predictions.csv", index=False)

    summary = (
        test.groupby(["dataset", "source_label", "wildguard_class"], dropna=False)
        .agg(
            samples=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence_percent", "mean"),
        )
        .reset_index()
    )
    summary["accuracy_percent"] = summary["accuracy"] * 100
    summary.to_csv(OUT / "source_summary.csv", index=False)

    report = classification_report(
        actuals, predictions, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    (OUT / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Device:", device)
    print("Test rows evaluated:", len(test))
    print("Accuracy:", round(float(test["correct"].mean()) * 100, 2), "%")
    print("\nSource summary:")
    print(summary.sort_values(["wildguard_class", "samples"], ascending=[True, False])
          .to_string(index=False))
    print("\nSaved diagnostic files in:", OUT)


if __name__ == "__main__":
    main()
