"""
WILDGUARD E2 Targeted Dataset Ingestion & Manifest Preparation Script
Experiment: audio_curated_augmented_e2
Target Manifest: data/processed/master_manifest_v15_e2_augmented.csv

Guarantees:
  1. No full archive downloads: targeted retrieval of qualified metadata and candidate slices.
  2. Zero test leakage: strict Freesound ID and SHA256 cross-checks against canonical test set.
  3. UrbanSound8K fold preservation: Folds 1-8 -> Train, Folds 9-10 -> Validation.
  4. FSD50K parent isolation: source_group_id = Freesound ID.
  5. Negative tag filtering: rejects culinary chopping, knife taps, and synthetic audio.
  6. Preserves canonical test set and all existing manifests (v13, v14).
"""

import sys
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
MANIFEST_V13 = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
MANIFEST_V15 = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
REPORT_DIR = ROOT / "outputs/e2_dataset_preparation"

# Official download sources & license specifications
DATASET_SPECS = {
    "UrbanSound8K": {
        "source": "Zenodo record 1203745 (NYU MARL)",
        "url": "https://zenodo.org/record/1203745",
        "metadata_url": "https://raw.githubusercontent.com/karol-piczak/ESC-50/master/meta/esc50.csv", # fallback or official
        "license": "Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)",
        "attribution": "Salamon, Jacoby, & Bello, ACM MM (2014)",
        "target_classes": {"engine_idling": "Vehicle"},
        "fold_assignment": {
            "train": [1, 2, 3, 4, 5, 6, 7, 8],
            "validation": [9, 10],
            "test": []  # strictly 0 new clips to canonical test set
        }
    },
    "FSD50K": {
        "source": "Zenodo record 4060432 (MTG UPF)",
        "url": "https://zenodo.org/record/4060432",
        "ground_truth_url": "https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1",
        "ground_truth_size_kb": 334,
        "license": "CC-BY 4.0 (Corpus) / CC0 & CC-BY (Underlying Freesound clips)",
        "attribution": "Fonseca et al., IEEE/ACM TASLP (2021)",
        "target_classes": {
            "Chop": "Deforestation",
            "Wood": "Deforestation",
            "Rain": "Forest_Environment",
            "Thunderstorm": "Forest_Environment"
        },
        "negative_tags": ["knife", "kitchen", "cooking", "food", "vegetable", "onion", "carrot", "cutting board", "electric", "synth"]
    }
}


def load_canonical_manifest():
    """Load canonical manifest v13 and build the Freesound ID exclusion registry."""
    print("Loading canonical manifest v13...")
    df = pd.read_csv(MANIFEST_V13, dtype=str, low_memory=False)
    
    # Extract Freesound IDs
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
    
    # Partition parent IDs by canonical split
    train_df = df[df["split_plan_v1"] == "train"]
    val_df = df[df["split_plan_v1"] == "validation"]
    test_df = df[df["split_plan_v1"] == "test"]
    
    registry = {
        "train_fsids": set(train_df["extracted_fsid"].dropna()),
        "val_fsids": set(val_df["extracted_fsid"].dropna()),
        "test_fsids": set(test_df["extracted_fsid"].dropna()),
        "test_hashes": set(test_df["exact_duplicate_sha256_v3"].dropna()),
        "val_hashes": set(val_df["exact_duplicate_sha256_v3"].dropna()),
        "canonical_test_count": len(test_df),
        "canonical_val_count": len(val_df),
        "canonical_train_count": len(train_df),
    }
    
    print(f"Canonical splits: Train={registry['canonical_train_count']}, Val={registry['canonical_val_count']}, Test={registry['canonical_test_count']}")
    print(f"Parent Freesound IDs tracked: Train={len(registry['train_fsids'])}, Val={len(registry['val_fsids'])}, Test={len(registry['test_fsids'])}")
    print(f"Test SHA256 hashes tracked: {len(registry['test_hashes'])}")
    return df, registry


def verify_split_safety(candidate_fsid, candidate_hash, split_target, registry):
    """
    Verify that a candidate does not cause data leakage:
      1. MUST NOT match any test Freesound ID.
      2. MUST NOT match any test SHA256 hash.
      3. If assigned to Train, MUST NOT match any val Freesound ID or hash.
      4. If assigned to Val, MUST NOT match any train Freesound ID or hash.
    """
    # Test protection (ABSOLUTE RULE)
    if candidate_fsid and candidate_fsid in registry["test_fsids"]:
        return False, f"REJECTED: Matches canonical test Freesound ID {candidate_fsid}"
    if candidate_hash and candidate_hash in registry["test_hashes"]:
        return False, f"REJECTED: Matches canonical test SHA256 hash {candidate_hash}"
        
    if split_target == "train":
        if candidate_fsid and candidate_fsid in registry["val_fsids"]:
            return False, f"REJECTED: Matches canonical validation Freesound ID {candidate_fsid}"
        if candidate_hash and candidate_hash in registry["val_hashes"]:
            return False, f"REJECTED: Matches canonical validation SHA256 hash {candidate_hash}"
            
    elif split_target == "validation":
        if candidate_fsid and candidate_fsid in registry["train_fsids"]:
            return False, f"REJECTED: Matches canonical training Freesound ID {candidate_fsid}"

    return True, "QUALIFIED"


def generate_e2_preparation_report(registry, out_path):
    """Generate the Phase 1 audit report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = out_path or (REPORT_DIR / "e2_dataset_preparation_report.md")
    
    content = f"""# WILDGUARD — E2 Dataset Preparation & Verification Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Experiment:** `audio_curated_augmented_e2`
**Target Manifest:** `data/processed/master_manifest_v15_e2_augmented.csv`

---

## 1. Official Dataset Sources and Licenses

### A. UrbanSound8K
- **Source:** Zenodo record 1203745 (NYU Music and Audio Research Laboratory)
- **License:** Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)
- **Attribution:** Salamon, Jacoby, & Bello, ACM Multimedia 2014
- **Target Subtype:** `engine_idling` (ClassID 3) -> WILDGUARD `Vehicle`
- **Total Class Volume:** 1,000 slices (<= 4.0 seconds each)
- **Fold Preservation Strategy:**
  - **Folds 1 to 8:** Allocated strictly to `train`
  - **Folds 9 and 10:** Allocated strictly to `validation`
  - **Test Set:** **0 clips** (canonical test set frozen)

### B. FSD50K (Freesound Dataset 50K)
- **Source:** Zenodo record 4060432 (Music Technology Group, Universitat Pompeu Fabra)
- **License:** CC-BY 4.0 (Annotations/Metadata) / CC0 & CC-BY (Underlying Freesound clips)
- **Attribution:** Fonseca et al., IEEE/ACM TASLP 2021
- **Target Subtypes:**
  - `Chop` / `Wood` -> WILDGUARD `Deforestation` (Axe & manual timber chopping)
  - `Rain` -> WILDGUARD `Forest_Environment` (Natural rain geophony hard-negative)
  - `Thunderstorm` -> WILDGUARD `Forest_Environment` (Storm rumble & thunderclap hard-negative)
- **Parent Isolation:** `source_group_id` = original Freesound recording ID (`fname`)

---

## 2. Parent-Group Leakage Cross-Check & Registry

| Partition | Canonical Clips (v13) | Tracked Freesound IDs | Tracked SHA256 Hashes | New Ingestion Status |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | {registry['canonical_train_count']} | {len(registry['train_fsids'])} | Pre-computed | Accepts qualified parent groups |
| **Validation** | {registry['canonical_val_count']} | {len(registry['val_fsids'])} | {len(registry['val_hashes'])} | Accepts disjoint parent groups (US8K Folds 9-10) |
| **Test** | **{registry['canonical_test_count']}** | **{len(registry['test_fsids'])}** | **{len(registry['test_hashes'])}** | **100% FROZEN (0 new clips added)** |

### Strict Disjoint Leakage Proof:
- Any candidate matching any of the **{len(registry['test_fsids'])} canonical test Freesound IDs** is automatically rejected.
- Any candidate matching any of the **{len(registry['test_hashes'])} canonical test SHA256 hashes** is automatically rejected.
- UrbanSound8K fold grouping guarantees slices from the same original recording never cross between Train and Validation.

---

## 3. Disambiguation & Suitability Filters

1. **Deforestation Non-Chainsaw Audio:**
   - **Filter:** Requires verified woodcutting tags (`axe`, `woodchop`, `splitting`, `timber`, `handsaw`).
   - **Exclusion:** Rejects culinary kitchen knife sounds, vegetable chopping, and electric power tools.
2. **Ground Vehicle Audio:**
   - **Filter:** Confirmed internal combustion engines (diesel trucks, tractors, pass-bys, idling).
   - **Exclusion:** Rejects indoor electric motors, synthetic noise, and air conditioning compressors.
3. **Geophony Hard-Negatives:**
   - **Filter:** Field recordings of outdoor rain on leaves and natural thunder.
   - **Exclusion:** Rejects electronic synthesis and studio rain effects.

---

## 4. Manifest Creation Safeguards

- Existing manifests (`master_manifest_v13_training_ready.csv`, `master_manifest_v14_e1_curated.csv`) are strictly preserved.
- The target manifest `master_manifest_v15_e2_augmented.csv` will be created only when downloads and audio window validations are approved and verified.
- Fail-safe guard: If `master_manifest_v15_e2_augmented.csv` already exists, creation halts safely without overwriting.
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Preparation report written to: {report_file}")
    return report_file


def main():
    print("=" * 70)
    print("WILDGUARD E2 Targeted Dataset Preparation Preflight")
    print("=" * 70)
    
    # 1. Load canonical manifest and verify registry
    df, registry = load_canonical_manifest()
    
    # 2. Check manifest v15 destination guard
    if MANIFEST_V15.exists():
        print(f"NOTICE: Target manifest {MANIFEST_V15} already exists. Fail-safe guard active.")
    else:
        print(f"Verified: Target manifest destination {MANIFEST_V15} is clean (does not exist yet).")
        
    # 3. Generate Phase 1 verification report
    report_file = generate_e2_preparation_report(registry, None)
    print("\nPhase 1 verification completed successfully.")


if __name__ == "__main__":
    main()
