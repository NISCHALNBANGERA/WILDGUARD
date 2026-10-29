
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
OUT = ROOT / "outputs/audio_balanced"

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
    audio, sr = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    audio = audio.mean(axis=1)

    if sr != SR:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=SR
        )

    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(
            audio, (0, N_SAMPLES - len(audio))
        )

    return audio


def make_features(path):
    audio = load_audio(path)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=40,
        fmax=SR // 2,
        center=False,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80, 0) / 80.0

    return torch.from_numpy(
        logmel.astype(np.float32)
    ).unsqueeze(0)


class AudioDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        x = make_features(row.audio_path)
        y = LABEL_TO_ID[row.wildguard_class]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default=str(OUT)
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    df = pd.read_csv(MANIFEST, dtype=str, low_memory=False)

    eligible = (
        df["is_training_eligible_v3"]
        .str.lower()
        .eq("true")
    )

    df = df.loc[eligible].copy()
    df = df[df["wildguard_class"].isin(CLASSES)].copy()

    df["split_plan_v1"] = (
        df["split_plan_v1"].str.lower().str.strip()
    )
    df["audio_path"] = df["audio_path"].astype(str)

    missing = [
        p for p in df["audio_path"]
        if not Path(p).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} eligible audio files are missing. "
            f"First missing path: {missing[0]}"
        )

    train = df[df["split_plan_v1"] == "train"].copy()
    val = df[df["split_plan_v1"] == "validation"].copy()
    test = df[df["split_plan_v1"] == "test"].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError("A required split is empty.")

    counts = train["wildguard_class"].value_counts()
    absent = [c for c in CLASSES if counts.get(c, 0) == 0]

    if absent:
        raise RuntimeError(
            f"Classes missing from training data: {absent}"
        )

    print(
        "Rows:", len(df),
        "| Train:", len(train),
        "| Validation:", len(val),
        "| Test:", len(test),
    )
    print("Classes:", CLASSES)
    print("Training class counts:\n", counts)

    model = SmallCNN(len(CLASSES)).to(device)

    sample_x, sample_y = AudioDataset(train.head(1))[0]

    with torch.no_grad():
        output = model(sample_x.unsqueeze(0).to(device))

    print("Feature shape:", tuple(sample_x.shape))
    print("Model output shape:", tuple(output.shape))
    print("Sample label:", CLASSES[sample_y])

    if args.smoke_test:
        print("SMOKE TEST PASSED. Full training has not started.")
        return

    out_dir = Path(args.out)

    # Overwrite guard for hard-negative experiment
    best_path = out_dir / "wildguard_audio_cnn_best.pt"
    if "hard_negative_experiment" in str(out_dir) and best_path.exists():
        print(f"TERMINATING: Output folder '{out_dir}' already contains a checkpoint.")
        print("To rerun the experiment, delete the existing output folder or use a different --out path.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

   
   
    # Experiment 2: softer class balancing, training split only.
    class_counts = train["wildguard_class"].value_counts()

    # Base class-balancing weights
    base_weights = train["wildguard_class"].map(lambda c: 1.0 / np.sqrt(class_counts[c]))

    # 2x multiplier for specific hard-negative source categories in training split
    hard_neg_mask = (
        ((train["dataset"] == "FSC22") & (train["source_label"].isin(["Silence", "Wind"]))) |
        ((train["dataset"] == "ESC-50") & (train["source_label"] == "insects"))
    )

    sample_weights = base_weights.copy()
    sample_weights[hard_neg_mask] *= 2.0
    sample_weights = sample_weights.to_numpy(dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train),
        replacement=True,
    )



    train_loader = DataLoader(
        AudioDataset(train),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        AudioDataset(val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    test_loader = DataLoader(
        AudioDataset(test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Sampling is balanced, so do not apply class weights again.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val = float("inf")
    best_path = out_dir / "wildguard_audio_cnn_best.pt"

    def evaluate(loader):
        model.eval()
        loss_sum = 0.0
        total = 0
        all_y = []
        all_pred = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                loss_sum += loss.item() * len(y)
                total += len(y)

                all_y.extend(y.cpu().numpy().tolist())
                all_pred.extend(
                    logits.argmax(1).cpu().numpy().tolist()
                )

        return loss_sum / max(1, total), all_y, all_pred

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        val_loss, _, _ = evaluate(val_loader)
        scheduler.step(val_loss)

        train_loss = loss_sum / max(1, total)
        train_acc = correct / max(1, total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.6g}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "sample_rate": SR,
                    "duration_seconds": DURATION,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP,
                },
                best_path,
            )

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, y_true, y_pred = evaluate(test_loader)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\nBest validation loss:", round(best_val, 4))
    print("Held-out test loss:", round(test_loss, 4))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(out_dir / "confusion_matrix.csv")

    with open(
        out_dir / "test_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("Best model:", best_path)
    print("Test report:", out_dir / "test_report.json")
    print("Confusion matrix:", out_dir / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
