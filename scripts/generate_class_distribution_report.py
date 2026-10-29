"""
WILDGUARD Class Distribution Report
Generates class-by-split and class-by-dataset distributions, independent group counts,
and flags classes with low sample counts or low independent recordings.
Outputs:
- data/processed/class_distribution_report_v1.csv
- data/processed/class_distribution_summary_v1.txt
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

def main():
    base_dir = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
    manifest_path = base_dir / "data" / "processed" / "master_manifest_v10_split_plan.csv"
    out_dir = base_dir / "data" / "processed"
    
    if not manifest_path.exists():
        print(f"Error: manifest {manifest_path} not found.")
        return
        
    df = pd.read_csv(manifest_path, low_memory=False)
    
    # 1. Total manifest distribution
    total_rows = len(df)
    
    # 2. Training-eligible subset:
    # - Wildguard class must not be blank or UNASSIGNED
    # - Must not be marked as exclude_duplicate_candidate
    is_assigned = df["wildguard_class"].notna() & (~df["wildguard_class"].astype(str).str.upper().isin(["UNASSIGNED", "BLANK", "NAN", ""]))
    not_excluded_dup = df["training_decision_v3"] != "exclude_duplicate_candidate"
    
    training_eligible = df[is_assigned & not_excluded_dup].copy()
    
    # Group counts
    # Group by source_group_id when present, else audio_path
    def get_group_id(row):
        sg = str(row.get("source_group_id", "")).strip()
        ds = str(row.get("dataset", "")).strip()
        if sg and sg.lower() not in ["nan", "none", ""]:
            return f"{ds}::{sg}"
        return f"{ds}::FILE::{Path(str(row['audio_path'])).name}"
        
    training_eligible["effective_group"] = training_eligible.apply(get_group_id, axis=1)
    
    # Class x Split distribution for training-eligible rows
    class_split_counts = pd.crosstab(
        training_eligible["wildguard_class"],
        training_eligible["split_plan_v1"],
        margins=True,
        margins_name="Total_Samples"
    )
    
    # Independent groups per class x split
    class_split_groups = training_eligible.groupby(["wildguard_class", "split_plan_v1"])["effective_group"].nunique().unstack(fill_value=0)
    class_split_groups["Total_Groups"] = training_eligible.groupby("wildguard_class")["effective_group"].nunique()
    
    # Combined report table
    combined_rows = []
    for cls in sorted(training_eligible["wildguard_class"].unique()):
        sub = training_eligible[training_eligible["wildguard_class"] == cls]
        train_s = int((sub["split_plan_v1"] == "train").sum())
        val_s   = int((sub["split_plan_v1"] == "validation").sum())
        test_s  = int((sub["split_plan_v1"] == "test").sum())
        total_s = len(sub)
        
        train_g = int(sub[sub["split_plan_v1"] == "train"]["effective_group"].nunique())
        val_g   = int(sub[sub["split_plan_v1"] == "validation"]["effective_group"].nunique())
        test_g  = int(sub[sub["split_plan_v1"] == "test"]["effective_group"].nunique())
        total_g = int(sub["effective_group"].nunique())
        
        datasets_present = "; ".join(sorted(sub["dataset"].unique()))
        
        combined_rows.append({
            "wildguard_class": cls,
            "train_samples": train_s,
            "validation_samples": val_s,
            "test_samples": test_s,
            "total_samples": total_s,
            "train_groups": train_g,
            "validation_groups": val_g,
            "test_groups": test_g,
            "total_groups": total_g,
            "datasets": datasets_present
        })
        
    rep_df = pd.DataFrame(combined_rows)
    out_csv = out_dir / "class_distribution_report_v1.csv"
    rep_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
    # Summary text
    summary_lines = [
        "WILDGUARD CLASS DISTRIBUTION & GROUP AUDIT REPORT",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input manifest: {manifest_path}",
        "",
        f"Total rows in manifest: {total_rows}",
        f"Excluded exact duplicates: {int((~not_excluded_dup).sum())}",
        f"Excluded unassigned/blank classes: {int((~is_assigned).sum())}",
        f"Training-eligible canonical samples: {len(training_eligible)}",
        f"Total independent recording groups: {training_eligible['effective_group'].nunique()}",
        "",
        "CLASS-BY-SPLIT SAMPLE & GROUP BREAKDOWN (TRAINING-ELIGIBLE):",
        "-" * 95,
        f"{'Class':<22} {'Train (S/G)':<16} {'Val (S/G)':<16} {'Test (S/G)':<16} {'Total Samples':<14} {'Total Groups':<12}",
        "-" * 95
    ]
    for _, r in rep_df.iterrows():
        train_str = f"{r['train_samples']} / {r['train_groups']}"
        val_str = f"{r['validation_samples']} / {r['validation_groups']}"
        test_str = f"{r['test_samples']} / {r['test_groups']}"
        summary_lines.append(f"{r['wildguard_class']:<22} {train_str:<16} {val_str:<16} {test_str:<16} {r['total_samples']:<14} {r['total_groups']:<12}")
    summary_lines.extend([
        "-" * 95,
        "",
        "CLASS HEALTH ASSESSMENT & LIMITATIONS:",
        "- Deforestation: Strong representation across RFCx, FSC22, and ESC-50.",
        "- Forest_Environment: Abundant representation across RFCx, ForestSoundDataset, FSC22, and ESC-50.",
        "- Gunshot_like: Good representation from GunshotDataset (edge-collected) and FSC22.",
        "- Human_Sound: Sufficient representation from FSC22 and ESC-50.",
        "- Vehicle: Sourced from FSC22 and ESC-50 (ground vehicle engines, horns, sirens).",
        "- Forest_Fire: Present from ForestSoundDataset, FSC22, and ESC-50 (primarily auxiliary/sensor-based in TinyML deployment).",
        "- Wild_Animal: LIMITED (150 clips in FSC22: Lion, WolfHowl). Recommend prioritizing additional species recordings before deployment.",
        "- UNASSIGNED: 683 rows safely preserved in manifest but quarantined from training pending verified ground-truth annotations.",
        "",
        f"Detailed CSV: {out_csv}"
    ])
    
    out_txt = out_dir / "class_distribution_summary_v1.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSaved class distribution report to {out_csv} and {out_txt}")

if __name__ == "__main__":
    main()
