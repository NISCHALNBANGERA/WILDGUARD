"""
WILDGUARD Dataset Audit Script
Scans all raw dataset directories and reports file counts, types, sizes.
Does NOT preprocess, move, or modify any files.
"""
import os
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent
RAW = BASE / "data" / "raw"

DATASETS = {
    "FSC22": {
        "path": RAW / "FSC22",
        "expected_audio": [".wav"],
        "metadata": RAW / "FSC22" / "extracted" / "Metadata-20220916T202011Z-001" / "Metadata" / "Metadata V1.0 FSC22.csv",
    },
    "ESC-50": {
        "path": RAW / "ESC-50" / "ESC-50-master",
        "expected_audio": [".wav"],
        "metadata": RAW / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv",
    },
    "RFCx_FrugalAI": {
        "path": RAW / "RFCx_FrugalAI",
        "expected_audio": [".wav"],
        "metadata": None,
    },
    "ForestWildFire": {
        "path": RAW / "ForestWildFire",
        "expected_audio": [".wav", ".mp3"],
        "metadata": None,
    },
}

def count_audio(root, exts):
    total = 0
    paths = []
    for ext in exts:
        found = list(root.rglob(f"*{ext}"))
        total += len(found)
        paths.extend(found)
    return total, paths

def dir_size_mb(root):
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)

print("=" * 70)
print("WILDGUARD DATASET AUDIT")
print("=" * 70)

for name, info in DATASETS.items():
    p = info["path"]
    print(f"\n--- {name} ---")
    if not p.exists():
        print(f"  STATUS: MISSING — path does not exist: {p}")
        continue

    audio_count, audio_paths = count_audio(p, info["expected_audio"])
    size_mb = dir_size_mb(p)
    print(f"  STATUS:       Present")
    print(f"  Path:         {p}")
    print(f"  Audio files:  {audio_count}")
    print(f"  Disk size:    {size_mb} MB")

    # Metadata
    meta = info.get("metadata")
    if meta and Path(meta).exists():
        df = pd.read_csv(meta)
        print(f"  Metadata CSV: Found ({len(df)} rows)")
        if name == "FSC22":
            print("  FSC22 Classes:")
            class_counts = df["Class Name"].value_counts()
            for cls, cnt in class_counts.items():
                print(f"    {cls:<20} {cnt} clips")
        elif name == "ESC-50":
            print("  ESC-50 selected classes relevant to WILDGUARD:")
            rel = ["chainsaw", "crackling_fire", "wind", "rain", "crickets",
                   "chirping_birds", "insects_flying", "dog", "wolf",
                   "helicopter", "engine", "train", "airplane", "gun_shot",
                   "fireworks", "clapping", "laughing", "crying_baby"]
            df_rel = df[df["category"].isin(rel)]
            print(f"    {df['category'].value_counts().to_string()}")
    elif meta:
        print(f"  Metadata CSV: NOT FOUND at {meta}")
    else:
        # RFCx — count by filename convention
        if name == "RFCx_FrugalAI":
            audio_dir = p / "audio"
            if audio_dir.exists():
                pos = list(audio_dir.glob("rfcx_pos_*.wav"))
                neg = list(audio_dir.glob("rfcx_neg_*.wav"))
                print(f"  RFCx label=chainsaw (pos): {len(pos)} files")
                print(f"  RFCx label=environment (neg): {len(neg)} files")
        if name == "ForestWildFire":
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    cnt = len(list(sub.rglob("*.wav"))) + len(list(sub.rglob("*.mp3")))
                    print(f"  Subdir '{sub.name}': {cnt} audio files")

print("\n" + "=" * 70)
print("WILDGUARD 7-CLASS COVERAGE FROM EXISTING DATASETS")
print("=" * 70)

CLASS_MAP = {
    "Forest_Environment": ["FSC22(Rain, Wind, Thunderstorm, WaterDrops, BirdChirping, Insect, Frog, Silence)",
                           "ESC-50(rain, wind, crickets, chirping_birds, insects_flying)"],
    "Forest_Fire":        ["FSC22(Fire)", "ForestWildFire(all WAVs)",
                           "ESC-50(crackling_fire)"],
    "Gunshot_like":       ["FSC22(Gunshot, Firework)"],
    "Deforestation":      ["FSC22(Chainsaw, Axe, Handsaw, WoodChop, TreeFalling)", "RFCx(chainsaw)"],
    "Human_Sound":        ["FSC22(Speaking, Footsteps, Clapping, Whistling)",
                           "ESC-50(clapping, laughing, crying_baby)"],
    "Vehicle":            ["FSC22(VehicleEngine)", "ESC-50(engine)"],
    "Wild_Animal":        ["FSC22(Lion, WolfHowl, Squirrel, WingFlaping)",
                           "ESC-50(dog, wolf) — PARTIAL"],
}

MISSING = {
    "Gunshot_like": "Only FSC22 (75 clips). Need more outdoor/real-world gunshot recordings.",
    "Wild_Animal":  "Missing: Elephant, Tiger, Leopard, Wild Boar, Bear. FSC22/ESC-50 have Lion, Wolf only.",
}

for cls, sources in CLASS_MAP.items():
    print(f"\n  {cls}:")
    for s in sources:
        print(f"    [+] {s}")
    if cls in MISSING:
        print(f"    [!] GAP: {MISSING[cls]}")

print("\n" + "=" * 70)
print("DATASETS REQUIRING ADDITIONAL DOWNLOAD")
print("=" * 70)
print("  1. Gunshot_like — outdoor/PAM gunshot datasets")
print("  2. Wild_Animal  — elephant, tiger, leopard, wild boar, bear, wolf recordings")
print("=" * 70)
