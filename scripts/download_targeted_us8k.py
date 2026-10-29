"""
WILDGUARD E2 — Targeted Retrieval of UrbanSound8K Ground Vehicle Audio
Retrieves qualified 'engine idling' audio slices from official folds.
Enforces strict test leakage checks and parent-recording isolation.
"""
import os
import io
import re
import wave
import hashlib
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
MANIFEST_V13 = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
OUT_DIR = ROOT / "data/raw/UrbanSound8K_targeted"
AUDIOS_DIR = OUT_DIR / "audios"

# HF base URL
HF_BASE = "https://huggingface.co/datasets/MahiA/UrbanSound8K/resolve/main/"

# Disjoint fold strategy
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
EXCLUDE_FOLDS = [10]  # reserved

def load_test_registry():
    """Load canonical test set Freesound IDs and SHA256 hashes."""
    print("Loading canonical test registry from v13...")
    df = pd.read_csv(MANIFEST_V13, dtype=str, low_memory=False)
    
    def extract_fsid(row):
        ds = row["dataset"]
        gid = str(row["source_group_id"])
        if ds == "ESC-50":
            return gid
        elif ds == "FSC22":
            m = re.match(r"^(\d+)", gid)
            if m:
                return m.group(1)
        return None

    df["extracted_fsid"] = df.apply(extract_fsid, axis=1)
    test_df = df[(df["is_training_eligible_v3"].str.lower() == "true") & (df["split_plan_v1"].str.lower() == "test")]
    
    test_fsids = set(test_df["extracted_fsid"].dropna())
    test_hashes = set(test_df["exact_duplicate_sha256_v3"].dropna())
    print(f"Test registry loaded: {len(test_fsids)} Freesound IDs, {len(test_hashes)} SHA256 hashes.")
    return test_fsids, test_hashes


def fetch_audio(item):
    """Download single audio slice and verify integrity."""
    rel_path, target_path, fsid = item
    if target_path.exists() and target_path.stat().st_size > 1000:
        with open(target_path, "rb") as f:
            content = f.read()
        sha = hashlib.sha256(content).hexdigest()
        return target_path, len(content), sha, "EXISTS"

    url = HF_BASE + rel_path
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(content)
        sha = hashlib.sha256(content).hexdigest()
        return target_path, len(content), sha, "DOWNLOADED"
    except Exception as e:
        return target_path, 0, "", f"ERROR: {e}"


def main():
    print("=" * 70)
    print("WILDGUARD E2 — UrbanSound8K Targeted Retrieval")
    print("=" * 70)

    test_fsids, test_hashes = load_test_registry()

    # 1. Fetch metadata
    print("Fetching UrbanSound8K metadata from repository...")
    dfs = []
    for split_csv in ["csv_files/train.csv", "csv_files/test.csv"]:
        req = urllib.request.Request(HF_BASE + split_csv, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            dfs.append(pd.read_csv(io.BytesIO(resp.read())))
    meta = pd.concat(dfs, ignore_index=True)
    
    engines = meta[meta["classname"] == "engine idling"].copy()
    print(f"Total engine idling records in metadata: {len(engines)}")

    # Parse fold, filename, fsid
    def parse_item(row):
        p = row["path"]
        parts = p.split("/")
        fold = int(parts[1].replace("fold", ""))
        fname = parts[2]
        fsid = fname.split("-")[0]
        return pd.Series([fold, fname, fsid])

    engines[["fold", "filename", "fsid"]] = engines.apply(parse_item, axis=1)

    # 2. Filter out any test matching Freesound IDs
    leak_mask = engines["fsid"].isin(test_fsids)
    print(f"Rejecting {leak_mask.sum()} slices due to matching canonical test Freesound IDs.")
    clean_engines = engines[~leak_mask].copy()

    # 3. Target candidate selection
    # Train: up to 30 per fold for folds 1-8 (~240 total)
    train_candidates = []
    for fold in TRAIN_FOLDS:
        sub = clean_engines[clean_engines["fold"] == fold]
        selected = sub.head(30)
        train_candidates.append(selected)
    train_df = pd.concat(train_candidates, ignore_index=True)
    train_df["e2_target_split"] = "train"

    # Validation: up to 50 from fold 9 (~50 total)
    val_sub = clean_engines[clean_engines["fold"] == 9]
    val_df = val_sub.head(50).copy()
    val_df["e2_target_split"] = "validation"

    candidates = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Selected candidates: Train={len(train_df)} (Folds 1-8), Validation={len(val_df)} (Fold 9), Total={len(candidates)}")

    # 4. Concurrently download
    items_to_fetch = []
    for _, row in candidates.iterrows():
        rel_path = row["path"]
        target_path = OUT_DIR / rel_path
        items_to_fetch.append((rel_path, target_path, row["fsid"]))

    print(f"Starting multi-threaded retrieval of {len(items_to_fetch)} audio files...")
    total_bytes = 0
    records = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_audio, items_to_fetch))

    download_count = 0
    exists_count = 0
    error_count = 0

    for (p, size, sha, status), (_, row) in zip(results, candidates.iterrows()):
        if status.startswith("ERROR"):
            error_count += 1
            print(f"Failed to fetch {row['path']}: {status}")
            continue
        elif status == "DOWNLOADED":
            download_count += 1
        elif status == "EXISTS":
            exists_count += 1

        total_bytes += size

        # Check against test hashes
        if sha in test_hashes:
            raise RuntimeError(f"CRITICAL: Downloaded file {p.name} matches test hash {sha}!")

        # Verify WAV format
        try:
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                frames = wf.getnframes()
                dur = frames / sr
        except Exception as e:
            print(f"WAV read error {p.name}: {e}")
            continue

        records.append({
            "rel_path": row["path"],
            "filename": row["filename"],
            "fold": row["fold"],
            "fsid": row["fsid"],
            "e2_target_split": row["e2_target_split"],
            "sha256": sha,
            "bytes": size,
            "sample_rate": sr,
            "channels": ch,
            "duration": dur,
            "file_path": str(p.resolve())
        })

    print(f"\nRetrieval complete: {download_count} downloaded, {exists_count} already cached, {error_count} errors.")
    print(f"Total downloaded size: {total_bytes / 1024 / 1024:.2f} MB")
    
    # Save retrieval manifest
    out_csv = OUT_DIR / "us8k_engine_idling_e2_retrieved.csv"
    retrieved_df = pd.DataFrame(records)
    retrieved_df.to_csv(out_csv, index=False)
    print(f"Retrieved manifest saved to: {out_csv} ({len(retrieved_df)} verified clips)")

if __name__ == "__main__":
    main()
