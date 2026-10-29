import shutil
import hashlib
from pathlib import Path
import pandas as pd

project = Path.cwd()
manifest = project / "data" / "processed" / "master_manifest_v13_training_ready.csv"
destination = Path.home() / "OneDrive" / "Desktop" / "final_training_dataset"

classes = [
    "Deforestation",
    "Forest_Environment",
    "Forest_Fire",
    "Gunshot_like",
    "Human_Sound",
    "Vehicle",
]

df = pd.read_csv(manifest, low_memory=False)

# Use the confirmed column positions from your manifest output.
path_col = df.columns[0]
class_col = df.columns[4]
eligible_col = df.columns[21]

eligible = df[eligible_col].astype(str).str.strip().str.lower().isin(["true", "1"])
selected = df[eligible & df[class_col].isin(classes)]

records = []
missing = 0
copied = 0

for _, row in selected.iterrows():
    source = Path(str(row[path_col]))
    label = str(row[class_col])

    if label not in classes or not source.is_file():
        missing += 1
        continue

    destination_class = destination / label
    destination_class.mkdir(parents=True, exist_ok=True)

    # Unique filename prevents collisions between datasets.
    suffix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    target = destination_class / f"{source.stem}_{suffix}{source.suffix}"

    if not target.exists():
        shutil.copy2(source, target)
        copied += 1

    records.append({
        "class": label,
        "source_dataset": row.iloc[2],
        "source_label": row.iloc[3],
        "original_path": str(source),
        "copied_path": str(target),
        "split": row.iloc[18],
    })

inventory = pd.DataFrame(records)
inventory_path = destination / "copied_files_manifest.csv"
inventory.to_csv(inventory_path, index=False)

print("\nCOPY COMPLETE")
print("Destination:", destination)
print("Eligible manifest rows:", len(selected))
print("New files copied:", copied)
print("Missing/unavailable files:", missing)
print("\nFiles per class:")
if not inventory.empty:
    print(inventory.groupby("class").size().reindex(classes, fill_value=0).to_string())
print("\nInventory:", inventory_path)
