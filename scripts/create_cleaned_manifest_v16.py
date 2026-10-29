"""
Create Cleaned Manifest v16 for WILDGUARD E3.
Fixes the 3 parent-source overlaps between Train and Validation:
Reassigns 3 UrbanSound8K validation slices (parent IDs: 176638, 119455, 186940)
to 'train' in split_plan_v1 so that parent recordings are strictly grouped in a single partition.
Zero modifications to the canonical frozen 8,189 test set.
"""

import os
import hashlib
import pandas as pd

PROJECT_ROOT = r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD"
os.chdir(PROJECT_ROOT)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    src_manifest = "data/processed/master_manifest_v15_e2_augmented.csv"
    dst_manifest = "data/processed/master_manifest_v16_e3_cleaned.csv"
    
    print(f"Loading source manifest: {src_manifest}")
    df = pd.read_csv(src_manifest, dtype=str, low_memory=False)
    orig_sha = compute_sha256(src_manifest)
    print(f"Source SHA256: {orig_sha}")
    print(f"Total rows: {len(df):,}")
    
    # Overlapping parent groups to resolve
    overlap_parents = {"176638", "119455", "186940"}
    mask = (df["source_group_id"].isin(overlap_parents)) & (df["split_plan_v1"] == "validation")
    reassign_count = mask.sum()
    print(f"Validation slices with parent overlap in train: {reassign_count}")
    
    df.loc[mask, "split_plan_v1"] = "train"
    df.loc[mask, "split_plan_note_v1"] = "Reassigned to train to prevent parent-source leakage with ESC-50"
    
    # Verification
    train_parents = set(df[df["split_plan_v1"] == "train"]["source_group_id"].dropna())
    val_parents = set(df[df["split_plan_v1"] == "validation"]["source_group_id"].dropna())
    test_parents = set(df[df["split_plan_v1"] == "test"]["source_group_id"].dropna())
    
    tv_overlap = train_parents.intersection(val_parents)
    tt_overlap = train_parents.intersection(test_parents)
    vt_overlap = val_parents.intersection(test_parents)
    
    print("\n--- Split Isolation Verification ---")
    print(f"Train/Val parent overlap:  {len(tv_overlap)}")
    print(f"Train/Test parent overlap: {len(tt_overlap)}")
    print(f"Val/Test parent overlap:   {len(vt_overlap)}")
    assert len(tv_overlap) == 0, f"Error: Train/Val overlap still exists: {tv_overlap}"
    assert len(tt_overlap) == 0, f"Error: Train/Test overlap exists: {tt_overlap}"
    assert len(vt_overlap) == 0, f"Error: Val/Test overlap exists: {vt_overlap}"
    
    # Canonical test set verification
    canonical_test = df[df["split"] == "test"]
    assert len(canonical_test) == 8189, f"Canonical test set changed! Count: {len(canonical_test)}"
    assert (df.loc[canonical_test.index, "split_plan_v1"] == "test").all(), "Canonical test samples not in test partition!"
    print(f"Canonical held-out test set: {len(canonical_test):,} samples (100% frozen and verified)")
    
    # Class support in validation
    val_df = df[
        (df["is_training_eligible_v3"].str.lower() == "true") &
        (df["split_plan_v1"].str.lower() == "validation")
    ]
    print(f"\nNew validation split count: {len(val_df):,} samples")
    print(val_df["wildguard_class"].value_counts())
    
    # Save new manifest
    df.to_csv(dst_manifest, index=False)
    new_sha = compute_sha256(dst_manifest)
    print(f"\nSaved cleaned manifest to: {dst_manifest}")
    print(f"New Manifest SHA256: {new_sha}")

if __name__ == "__main__":
    main()
