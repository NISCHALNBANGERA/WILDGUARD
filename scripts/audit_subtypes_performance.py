"""
WILDGUARD Subtype Performance Audit (Batched)
Evaluates checkpoint: outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt
across all individual subtypes in validation and test splits using DataLoader for high throughput.
"""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

root = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
sys.path.append(str(root / "scripts"))
from train_wildguard_audio import AudioDataset, SmallCNN, CLASSES

manifest_path = root / "data/processed/master_manifest_v13_training_ready.csv"
df = pd.read_csv(manifest_path, dtype=str, low_memory=False)

eligible = df[df["is_training_eligible_v3"].str.lower() == "true"].copy()
target_classes = ["Deforestation", "Forest_Environment", "Forest_Fire", "Gunshot_like", "Human_Sound", "Vehicle"]
eligible = eligible[eligible["wildguard_class"].isin(target_classes)].copy()

ckpt_path = root / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model = SmallCNN(len(CLASSES))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running subtype audit on device: {device}")

def evaluate_split(split_name):
    sub_df = eligible[eligible["split_plan_v1"].str.lower() == split_name].copy().reset_index(drop=True)
    print(f"\n=======================================================")
    print(f"EVALUATING SUBTYPES ON {split_name.upper()} SPLIT ({len(sub_df)} samples)")
    print(f"=======================================================")
    
    loader = torch.utils.data.DataLoader(AudioDataset(sub_df), batch_size=64, shuffle=False, num_workers=0)
    
    all_preds = []
    t0 = time.time()
    with torch.no_grad():
        for b_idx, (x, _) in enumerate(loader):
            logits = model(x.to(device))
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            if (b_idx + 1) % 40 == 0:
                print(f"  Processed {len(all_preds)}/{len(sub_df)} in {time.time()-t0:.1f}s...")
                
    sub_df["pred_class"] = [CLASSES[i] for i in all_preds]
    
    records = []
    for (cls, ds, sl), g in sub_df.groupby(["wildguard_class", "dataset", "source_label"]):
        total = len(g)
        correct = (g["pred_class"] == cls).sum()
        acc = correct / total if total > 0 else 0.0
        
        pred_counts = g["pred_class"].value_counts()
        top_pred = pred_counts.index[0]
        top_pred_pct = pred_counts.iloc[0] / total * 100
        
        confused = g[g["pred_class"] != cls]["pred_class"]
        if not confused.empty:
            conf_counts = confused.value_counts()
            main_conf = f"{conf_counts.index[0]} ({conf_counts.iloc[0]}/{total})"
        else:
            main_conf = "None (100% correct)"
            
        records.append({
            "split": split_name,
            "class": cls,
            "dataset": ds,
            "subtype": sl,
            "samples": total,
            "accuracy": round(acc * 100, 1),
            "top_prediction": f"{top_pred} ({top_pred_pct:.0f}%)",
            "main_confusion": main_conf
        })
        
    res_df = pd.DataFrame(records)
    print(res_df.to_string(index=False))
    return res_df

val_results = evaluate_split("validation")
test_results = evaluate_split("test")

# Save outputs
out_dir = root / "outputs/subclass_data_audit_v1"
out_dir.mkdir(parents=True, exist_ok=True)
val_results.to_csv(out_dir / "subtype_performance_validation.csv", index=False)
test_results.to_csv(out_dir / "subtype_performance_test.csv", index=False)
print(f"\nSaved subtype evaluation tables to: {out_dir}")
