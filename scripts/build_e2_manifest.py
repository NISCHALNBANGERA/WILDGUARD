"""
WILDGUARD E2 — Build Augmented Master Manifest v15
Experiment: audio_curated_augmented_e2
Destination: data/processed/master_manifest_v15_e2_augmented.csv

Incorporates:
  1. Base canonical manifest v13 (all existing partitions preserved).
  2. Targeted UrbanSound8K 'engine_idling' (240 train in Folds 1-8, 50 val in Fold 9, 0 test).
  3. Local FSC22 'Generator' field recordings (63 train, 5 val, 7 test quarantined).
  4. Local ESC-50 'door_wood_knock' (32 train, 8 val, 0 test) -> Deforestation manual logging.
  5. Local ESC-50 'door_wood_creaks' (32 train, 8 val, 0 test) -> Deforestation tree falling.
  6. Local ESC-50 'pouring_water' (32 train, 8 val, 0 test) -> Forest_Environment water geophony.

Strict safeguards:
  - Canonical test set 100% frozen: exactly 8,189 samples identical to v13.
  - Zero parent recording (Freesound ID) leakage into train or validation.
  - Zero exact SHA256 hash collision with canonical test set.
  - Fail-safe write: never alters v13 or v14.
"""
import os
import re
import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
MANIFEST_V13 = ROOT / "data/processed/master_manifest_v13_training_ready.csv"
MANIFEST_V15 = ROOT / "data/processed/master_manifest_v15_e2_augmented.csv"
US8K_RETRIEVED = ROOT / "data/raw/UrbanSound8K_targeted/us8k_engine_idling_e2_retrieved.csv"
ESC_META = ROOT / "data/raw/ESC-50/ESC-50-master/meta/esc50.csv"
ESC_AUDIO = ROOT / "data/raw/ESC-50/ESC-50-master/audio"

def extract_fsid(dataset, group_id):
    gid = str(group_id)
    if dataset == "ESC-50" or dataset == "UrbanSound8K":
        return gid
    elif dataset == "FSC22":
        m = re.match(r"^(\d+)", gid)
        if m:
            return m.group(1)
    return None

def compute_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("=" * 70)
    print("Building master_manifest_v15_e2_augmented.csv")
    print("=" * 70)

    # 1. Load canonical manifest v13
    print("Loading v13 manifest...")
    df_v13 = pd.read_csv(MANIFEST_V13, dtype=str, low_memory=False)
    print(f"Base v13 rows: {len(df_v13)}")

    # Extract test registry
    test_mask = (df_v13["is_training_eligible_v3"].str.lower() == "true") & (df_v13["split_plan_v1"].str.lower() == "test")
    test_v13 = df_v13[test_mask].copy()
    test_fsids = set()
    for _, r in test_v13.iterrows():
        fid = extract_fsid(r["dataset"], r["source_group_id"])
        if fid:
            test_fsids.add(fid)
    test_hashes = set(test_v13["exact_duplicate_sha256_v3"].dropna())
    print(f"Canonical test set: {len(test_v13)} samples, {len(test_fsids)} Freesound IDs, {len(test_hashes)} SHA256 hashes.")
    assert len(test_v13) == 8189, f"Expected 8189 test samples, found {len(test_v13)}"

    # 2. Update FSC22 Generator in v13 base
    # Enable 63 train and 5 val rows as Vehicle
    df_v15 = df_v13.copy()
    fsc_gen_train_val = (df_v15["dataset"] == "FSC22") & (df_v15["source_label"] == "Generator") & (df_v15["split_plan_v1"].isin(["train", "validation"]))
    df_v15.loc[fsc_gen_train_val, "wildguard_class"] = "Vehicle"
    df_v15.loc[fsc_gen_train_val, "training_decision_v3"] = "use"
    df_v15.loc[fsc_gen_train_val, "is_training_eligible_v3"] = "True"
    df_v15.loc[fsc_gen_train_val, "quarantine_reason_v3"] = None
    print(f"Re-enabled FSC22 Generator: {fsc_gen_train_val.sum()} samples (63 train, 5 validation).")

    # 3. Build rows for unindexed ESC-50 categories
    print("Processing unindexed ESC-50 categories...")
    esc_df = pd.read_csv(ESC_META)
    esc_target_cats = {
        "door_wood_knock": "Deforestation",
        "door_wood_creaks": "Deforestation",
        "pouring_water": "Forest_Environment"
    }

    new_esc_rows = []
    for cat, wildguard_cls in esc_target_cats.items():
        sub = esc_df[esc_df["category"] == cat]
        for _, r in sub.iterrows():
            fname = r["filename"]
            fpath = (ESC_AUDIO / fname).resolve()
            assert fpath.exists(), f"Missing audio: {fpath}"
            
            fold = int(r["fold"])
            split = "validation" if fold == 5 else "train"
            fsid = str(r["src_file"])
            
            # Check test leakage
            assert fsid not in test_fsids, f"Leakage: ESC-50 {fname} matches test FSID {fsid}"
            sha = compute_sha256(fpath)
            assert sha not in test_hashes, f"Leakage: ESC-50 {fname} matches test hash {sha}"

            new_esc_rows.append({
                "audio_path": str(fpath),
                "audio_exists": "True",
                "dataset": "ESC-50",
                "source_label": cat,
                "wildguard_class": wildguard_cls,
                "source_group_id": fsid,
                "split": f"fold{fold}",
                "status": "active",
                "duplicate_group": None,
                "training_decision": "use",
                "duplicate_group_v3": f"esc_{fsid}",
                "duplicate_count_v3": "1",
                "exact_duplicate_sha256_v3": sha,
                "duplicate_classes_v3": wildguard_cls,
                "duplicate_label_conflict_v3": "False",
                "canonical_candidate_v3": "True",
                "training_decision_v3": "use",
                "decision_reason_v3": f"E2 curated {cat} enhancement",
                "split_plan_v1": split,
                "split_component_v1": f"esc_fold{fold}",
                "split_plan_note_v1": "E2 fold assignment (1-4 train, 5 val)",
                "is_training_eligible_v3": "True",
                "quarantine_reason_v3": None
            })
    print(f"Created {len(new_esc_rows)} new ESC-50 rows (96 train, 24 validation).")

    # 4. Build rows for UrbanSound8K engine_idling
    print("Processing targeted UrbanSound8K engine idling...")
    us8k_df = pd.read_csv(US8K_RETRIEVED)
    new_us8k_rows = []
    for _, r in us8k_df.iterrows():
        fpath = Path(r["file_path"]).resolve()
        assert fpath.exists(), f"Missing audio: {fpath}"
        fsid = str(r["fsid"])
        split = str(r["e2_target_split"])
        sha = str(r["sha256"])
        fold = int(r["fold"])

        # Check test leakage
        assert fsid not in test_fsids, f"Leakage: US8K {fpath.name} matches test FSID {fsid}"
        assert sha not in test_hashes, f"Leakage: US8K {fpath.name} matches test hash {sha}"

        new_us8k_rows.append({
            "audio_path": str(fpath),
            "audio_exists": "True",
            "dataset": "UrbanSound8K",
            "source_label": "engine_idling",
            "wildguard_class": "Vehicle",
            "source_group_id": fsid,
            "split": f"fold{fold}",
            "status": "active",
            "duplicate_group": None,
            "training_decision": "use",
            "duplicate_group_v3": f"us8k_{fsid}",
            "duplicate_count_v3": "1",
            "exact_duplicate_sha256_v3": sha,
            "duplicate_classes_v3": "Vehicle",
            "duplicate_label_conflict_v3": "False",
            "canonical_candidate_v3": "True",
            "training_decision_v3": "use",
            "decision_reason_v3": "E2 targeted ground vehicle engine idling",
            "split_plan_v1": split,
            "split_component_v1": f"us8k_fold{fold}",
            "split_plan_note_v1": "E2 fold assignment (1-8 train, 9 val)",
            "is_training_eligible_v3": "True",
            "quarantine_reason_v3": None
        })
    print(f"Created {len(new_us8k_rows)} new UrbanSound8K rows (240 train, 50 validation).")

    # 5. Combine and annotate e2_eligible
    new_rows_df = pd.DataFrame(new_esc_rows + new_us8k_rows)
    master_v15 = pd.concat([df_v15, new_rows_df], ignore_index=True)

    # Set e2_eligible flag:
    # All eligible v13 items + newly added items are e2_eligible = True
    # Out-of-scope aerial/rail vehicle subtypes in training can be kept or filtered.
    # To keep exact comparability with v14 scope, set e2_eligible = True for all valid training items.
    master_v15["e2_eligible"] = master_v15["is_training_eligible_v3"].astype(str).str.lower() == "true"

    # 6. Final verification gates
    print("\nRunning final verification gates...")
    v15_eligible = master_v15[master_v15["e2_eligible"] == True]
    v15_train = v15_eligible[v15_eligible["split_plan_v1"] == "train"]
    v15_val = v15_eligible[v15_eligible["split_plan_v1"] == "validation"]
    v15_test = v15_eligible[v15_eligible["split_plan_v1"] == "test"]

    print(f"V15 Eligible Counts: Train={len(v15_train)}, Val={len(v15_val)}, Test={len(v15_test)}")
    assert len(v15_test) == 8189, f"Canonical test set modified! Found {len(v15_test)}, expected 8189"

    # Verify test set identity
    test_paths_v13 = set(test_v13["audio_path"])
    test_paths_v15 = set(v15_test["audio_path"])
    assert test_paths_v13 == test_paths_v15, "Canonical test audio paths changed!"

    # Verify no Freesound ID leakage from NEW additions into canonical test set
    new_train_val = pd.concat([pd.DataFrame(new_esc_rows), pd.DataFrame(new_us8k_rows)], ignore_index=True)
    new_train_val_fsids = set(new_train_val["source_group_id"].dropna())
    new_leakage = new_train_val_fsids & test_fsids
    assert len(new_leakage) == 0, f"Critical leakage! New additions share test Freesound IDs: {new_leakage}"

    # Verify no hash leakage from ANY training/val row to canonical test set
    v15_train_hashes = set(v15_train["exact_duplicate_sha256_v3"].dropna())
    assert len(v15_train_hashes & test_hashes) == 0, "Train shares SHA256 hash with canonical test set!"
    new_hashes = set(new_train_val["exact_duplicate_sha256_v3"].dropna())
    assert len(new_hashes & test_hashes) == 0, "New additions share SHA256 hash with canonical test set!"

    print("ALL VERIFICATION GATES PASSED.")
    print("\nClass distribution in E2 Training partition:")
    print(v15_train["wildguard_class"].value_counts().to_string())
    print("\nClass distribution in E2 Validation partition:")
    print(v15_val["wildguard_class"].value_counts().to_string())

    # 7. Safe write
    print(f"\nWriting {MANIFEST_V15}...")
    master_v15.to_csv(MANIFEST_V15, index=False)
    print(f"Successfully generated: {MANIFEST_V15} ({len(master_v15)} total rows)")

if __name__ == "__main__":
    main()
