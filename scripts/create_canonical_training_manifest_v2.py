from pathlib import Path
from datetime import datetime
import pandas as pd

BASE = Path.cwd()
INPUT = BASE / "data/processed/master_manifest_v10_split_plan_gunshot_mapping_candidate.csv"
OUT = BASE / "data/processed"
MANIFEST = OUT / "master_manifest_v12_training_canonical.csv"
SUMMARY = OUT / "training_canonical_summary_v2.txt"

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)
    original_rows = len(df)

    required = [
        "audio_path", "audio_exists", "dataset", "source_label",
        "wildguard_class", "training_decision_v3",
        "canonical_candidate_v3", "split_plan_v1",
        "source_group_id", "exact_duplicate_sha256_v3"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cls = df["wildguard_class"].fillna("").astype(str).str.strip()
    decision = df["training_decision_v3"].fillna("").astype(str).str.strip()
    split = df["split_plan_v1"].fillna("").astype(str).str.strip().str.lower()
    candidate = df["canonical_candidate_v3"].astype(str).str.lower().isin(["true", "1", "yes"])
    audio_ok = df["audio_exists"].astype(str).str.lower().isin(["true", "1", "yes"])
    allowed_splits = split.isin(["train", "validation", "test"])
    excluded_decisions = decision.isin([
        "exclude_duplicate_candidate",
        "manual_unassigned_review",
        "review_pending",
        "candidate_review_pending"
    ])
    assigned = (~cls.str.upper().isin(["", "NAN", "NONE", "UNASSIGNED", "BLANK"]))
    not_wild_animal = cls.ne("Wild_Animal")

    # Rebuild effective_group_id from the source manifest fields.
    source_group = df["source_group_id"].fillna("").astype(str).str.strip()
    dataset = df["dataset"].fillna("").astype(str).str.strip()
    filename = df["audio_path"].fillna("").astype(str).map(lambda p: Path(p).name)
    df["effective_group_id"] = [
        f"{ds}::{sg}" if sg and sg.lower() not in ["nan", "none", ""]
        else f"{ds}::FILE::{fn}"
        for ds, sg, fn in zip(dataset, source_group, filename)
    ]
    # Preserve split plan and all rows; eligibility is stricter than class assignment.
    eligible = (
        assigned & candidate & audio_ok & allowed_splits &
        (~excluded_decisions) & not_wild_animal
    )

    df["is_training_eligible_v2"] = eligible
    df["quarantine_reason_v2"] = "ELIGIBLE_CANONICAL_TRAINING_SAMPLE"
    df.loc[~assigned, "quarantine_reason_v2"] = "UNASSIGNED_CLASS"
    df.loc[~candidate, "quarantine_reason_v2"] = "NOT_CANONICAL_CANDIDATE"
    df.loc[~audio_ok, "quarantine_reason_v2"] = "AUDIO_NOT_AVAILABLE"
    df.loc[~allowed_splits, "quarantine_reason_v2"] = "INVALID_OR_MISSING_SPLIT"
    df.loc[decision.eq("exclude_duplicate_candidate"), "quarantine_reason_v2"] = "EXACT_DUPLICATE_EXCLUDED"
    df.loc[decision.eq("manual_unassigned_review"), "quarantine_reason_v2"] = "MANUAL_REVIEW_REQUIRED"
    df.loc[decision.eq("review_pending"), "quarantine_reason_v2"] = "REVIEW_PENDING"
    df.loc[decision.eq("candidate_review_pending"), "quarantine_reason_v2"] = "CANDIDATE_REVIEW_PENDING"
    df.loc[cls.eq("Wild_Animal"), "quarantine_reason_v2"] = "EXCLUDED_FROM_BASELINE_WILD_ANIMAL"

    # Ensure no exact duplicate hash contributes more than one eligible row.
    dup_hash = df["exact_duplicate_sha256_v3"].fillna("").astype(str).str.strip()
    dup_hash = dup_hash.mask(dup_hash.str.lower().isin(["", "nan", "none"]))
    duplicate_eligible = df[eligible & dup_hash.notna()]
    duplicated_hashes = duplicate_eligible["exact_duplicate_sha256_v3"].duplicated(keep="first")
    if duplicated_hashes.any():
        rows_to_exclude = duplicate_eligible.index[duplicated_hashes]
        df.loc[rows_to_exclude, "is_training_eligible_v2"] = False
        df.loc[rows_to_exclude, "quarantine_reason_v2"] = "EXACT_DUPLICATE_HASH_ALREADY_INCLUDED"

    # Recheck group/hash split leakage across the full proposed split plan.
    for col in ["effective_group_id", "exact_duplicate_sha256_v3"]:
        values = df[col].fillna("").astype(str).str.strip()
        valid = ~values.str.lower().isin(["", "nan", "none"])
        check = df.loc[valid, [col, "split_plan_v1"]].copy()
        leaks = check.groupby(col)["split_plan_v1"].nunique()
        if (leaks > 1).any():
            examples = leaks[leaks > 1].index[:5].tolist()
            raise ValueError(f"{col} crosses split boundaries; examples: {examples}")

    df.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    eligible_df = df[df["is_training_eligible_v2"]]
    counts_split = eligible_df["split_plan_v1"].value_counts()
    counts_class = eligible_df["wildguard_class"].value_counts()

    summary = [
        "WILDGUARD CORRECTED CANONICAL MANIFEST SUMMARY",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {INPUT}",
        f"Output: {MANIFEST}",
        f"Rows preserved: {original_rows}",
        f"Eligible baseline samples: {len(eligible_df)}",
        f"Excluded rows: {original_rows - len(eligible_df)}",
        "",
        "Eligible by split:"
    ]
    for s in ["train", "validation", "test"]:
        summary.append(f"  {s}: {int(counts_split.get(s, 0))}")
    summary.extend(["", "Eligible by class:"])
    for c, n in sorted(counts_class.items()):
        summary.append(f"  {c}: {n}")
    summary.extend([
        "",
        "NOTE: Manifest metadata and split integrity do not verify audio semantics.",
        "Gunshot_like labels from source_label=poaching require dataset/source validation."
    ])
    text = "\n".join(summary) + "\n"
    SUMMARY.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {MANIFEST}")
    print(f"Saved: {SUMMARY}")

if __name__ == "__main__":
    main()


