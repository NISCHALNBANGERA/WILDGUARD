"""
WILDGUARD Canonical Training Manifest Generator
Phase 2D & Phase 3 - Produces master_manifest_v11_training_canonical.csv
Preserves all 58,291 original rows, adds verified training eligibility flags,
quarantine reasons, and group-safe split assignments.
Outputs:
- data/processed/master_manifest_v11_training_canonical.csv
- data/processed/training_canonical_summary_v1.txt
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

def main():
    base_dir = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
    manifest_in = base_dir / "data" / "processed" / "master_manifest_v10_split_plan.csv"
    out_dir = base_dir / "data" / "processed"
    
    if not manifest_in.exists():
        print(f"Error: {manifest_in} not found.")
        return
        
    print(f"Reading input manifest: {manifest_in}")
    df = pd.read_csv(manifest_in, low_memory=False)
    total_rows = len(df)
    
    # 1. Determine training eligibility
    is_unassigned = df["wildguard_class"].isna() | df["wildguard_class"].astype(str).str.upper().isin(["UNASSIGNED", "BLANK", "NAN", ""])
    is_dup_excluded = df["training_decision_v3"] == "exclude_duplicate_candidate"
    
    eligible = (~is_unassigned) & (~is_dup_excluded)
    
    quarantine_reasons = []
    for idx, row in df.iterrows():
        if is_dup_excluded.iloc[idx]:
            reason = "EXACT_DUPLICATE_OF_CANONICAL_SAMPLE"
        elif is_unassigned.iloc[idx]:
            ds = str(row.get("dataset", ""))
            lbl = str(row.get("source_label", ""))
            reason = f"UNASSIGNED_CLASS_QUARANTINE::{ds}_{lbl}"
        else:
            reason = "ELIGIBLE_CANONICAL_TRAINING_SAMPLE"
        quarantine_reasons.append(reason)
        
    df["is_training_eligible"] = eligible
    df["quarantine_reason_v1"] = quarantine_reasons
    
    # 2. Build effective_group_id for all rows
    effective_groups = []
    for idx, row in df.iterrows():
        sg = str(row.get("source_group_id", "")).strip()
        ds = str(row.get("dataset", "")).strip()
        if sg and sg.lower() not in ["nan", "none", ""]:
            effective_groups.append(f"{ds}::{sg}")
        else:
            effective_groups.append(f"{ds}::FILE::{Path(str(row['audio_path'])).name}")
    df["effective_group_id"] = effective_groups
    
    # 3. Output paths
    out_manifest = out_dir / "master_manifest_v11_training_canonical.csv"
    out_summary = out_dir / "training_canonical_summary_v1.txt"
    
    df.to_csv(out_manifest, index=False, encoding="utf-8-sig")
    print(f"Saved canonical training manifest to {out_manifest} ({len(df)} rows)")
    
    eligible_df = df[df["is_training_eligible"]]
    counts_by_split = eligible_df["split_plan_v1"].value_counts().to_dict()
    counts_by_class = eligible_df["wildguard_class"].value_counts().to_dict()
    
    summary = f"""WILDGUARD CANONICAL TRAINING MANIFEST SUMMARY
Generated: {datetime.now().isoformat(timespec='seconds')}
Source: {manifest_in}
Output: {out_manifest}

SUMMARY STATISTICS:
- Total rows preserved: {total_rows}
- Training-eligible canonical samples: {len(eligible_df)} ({len(eligible_df)/total_rows*100:.2f}%)
- Quarantined exact duplicates: {int(is_dup_excluded.sum())} ({int(is_dup_excluded.sum())/total_rows*100:.2f}%)
- Quarantined unassigned classes: {int(is_unassigned.sum())} ({int(is_unassigned.sum())/total_rows*100:.2f}%)
- Independent recording groups in training set: {eligible_df['effective_group_id'].nunique()}

TRAINING-ELIGIBLE SAMPLES BY SPLIT:
- train:      {counts_by_split.get('train', 0):>6} ({counts_by_split.get('train', 0)/len(eligible_df)*100:.2f}%)
- validation: {counts_by_split.get('validation', 0):>6} ({counts_by_split.get('validation', 0)/len(eligible_df)*100:.2f}%)
- test:       {counts_by_split.get('test', 0):>6} ({counts_by_split.get('test', 0)/len(eligible_df)*100:.2f}%)

TRAINING-ELIGIBLE SAMPLES BY CLASS:
"""
    for cls, cnt in sorted(counts_by_class.items()):
        summary += f"- {cls:<22} {cnt:>6}\n"
        
    out_summary.write_text(summary, encoding="utf-8")
    print(f"Saved summary to {out_summary}")
    print("\n" + summary)

if __name__ == "__main__":
    main()
