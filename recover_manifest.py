
import hashlib
from pathlib import Path
import pandas as pd

project = Path(__file__).resolve().parent
master = project / "data" / "processed" / "master_manifest_v13_training_ready.csv"
dataset = Path.home() / "OneDrive" / "Desktop" / "dataset"

classes = [
    "Deforestation", "Forest_Environment", "Forest_Fire",
    "Gunshot_like", "Human_Sound", "Vehicle",
]

df = pd.read_csv(master, low_memory=False)
path_col = df.columns[0]
class_col = df.columns[4]
eligible_col = df.columns[21]

eligible = (
    df[eligible_col].astype(str).str.strip().str.lower()
    .isin(["true", "1"])
)
selected = df[eligible & df[class_col].isin(classes)]

records = []
missing = []
ambiguous = []

for _, row in selected.iterrows():
    source_text = str(row[path_col])
    source = Path(source_text)
    cls = str(row[class_col])

    suffix = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()[:12]

    target = dataset / cls / f"{source.stem}_{suffix}{source.suffix}"

    if target.is_file():
        records.append({
            "class": cls,
            "source_dataset": row.iloc[2],
            "source_label": row.iloc[3],
            "original_path": source_text,
            "copied_path": str(target.resolve()),
            "split": row.iloc[18],
        })
    else:
        missing.append(str(target))

print("Eligible master rows:", len(selected))
print("Matched existing audio files:", len(records))
print("Unmatched expected files:", len(missing))
print("Ambiguous matches:", len(ambiguous))

if records:
    result = pd.DataFrame(records)
    print("\nMatched files by class and split:")
    print(pd.crosstab(result["class"], result["split"]).to_string())

    output = dataset / "copied_files_manifest_recovered.csv"
    result.to_csv(output, index=False)
    print("\nRecovered manifest:", output)
else:
    print("\nNo files matched. No manifest written.")

