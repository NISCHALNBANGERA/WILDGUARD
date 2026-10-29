import pandas as pd
import numpy as np

ROOT = 'C:/Users/Nischal N Bangera/OneDrive/Desktop/WILDGUARD'
df = pd.read_csv(f'{ROOT}/data/processed/master_manifest_v16_e3_cleaned.csv', dtype=str, low_memory=False)
train_df = df[(df['is_training_eligible_v3'].str.lower() == 'true') & (df['split_plan_v1'].str.lower() == 'train')].copy()

total_samples = len(train_df)

# 1. Baseline Sampler (from train_wildguard_audio.py)
class_counts = train_df['wildguard_class'].value_counts()
base_w = train_df['wildguard_class'].map(lambda c: 1.0 / np.sqrt(class_counts[c]))
base_sample_w = base_w.copy()
hard_neg_mask_base = (
    ((train_df['dataset'] == 'FSC22') & (train_df['source_label'].isin(['Silence', 'Wind']))) |
    ((train_df['dataset'] == 'ESC-50') & (train_df['source_label'] == 'insects'))
)
base_sample_w[hard_neg_mask_base] *= 2.0

# 2. E3.4 Sampler (from train_e3_4_targeted_hn.py)
e3_4_w = base_w.copy()
gunshot_cap = 1.0 / np.sqrt(class_counts['Gunshot_like'])
e3_4_w[train_df['wildguard_class'] == 'Human_Sound'] = gunshot_cap
e3_4_w[train_df['wildguard_class'] == 'Vehicle'] = gunshot_cap

rumble_mask = (
    (train_df['wildguard_class'] == 'Forest_Environment') &
    (train_df['source_label'].astype(str).str.lower().isin(['thunderstorm', 'wind', 'sea_waves', 'rain', 'silence', 'water_drops', 'water']))
)
e3_4_w[rumble_mask] *= 2.5

logging_mask = (
    (train_df['wildguard_class'] == 'Deforestation') &
    (train_df['source_label'].isin(['Axe', 'WoodChop', 'TreeFalling', 'Handsaw', 'hand_saw', 'door_wood_knock', 'door_wood_creaks']))
)
e3_4_w[logging_mask] *= 1.5

# 3. E3.5 Proposed Sampler (Human Uncapped)
e3_5_w = base_w.copy()
e3_5_w[train_df['wildguard_class'] == 'Vehicle'] = gunshot_cap
e3_5_w[rumble_mask] *= 2.5
e3_5_w[logging_mask] *= 1.5

def audit_sampler(name, weights):
    total_w = weights.sum()
    probs = weights / total_w
    
    print(f"=== Sampler: {name} ===")
    print(f"Total Weight Sum: {total_w:.2f}")
    header = f"{'Class':<20} | {'Count':<7} | {'Mean Weight':<12} | {'Expected Draws':<16} | {'Draw %':<8} | {'Over/Under Ratio':<16}"
    print(header)
    print("-" * len(header))
    for c in ['Deforestation', 'Forest_Environment', 'Forest_Fire', 'Gunshot_like', 'Human_Sound', 'Vehicle']:
        c_mask = train_df['wildguard_class'] == c
        c_count = c_mask.sum()
        c_prob = probs[c_mask].sum()
        expected_draws = c_prob * total_samples
        mean_w = weights[c_mask].mean()
        ratio = expected_draws / c_count
        print(f"{c:<20} | {c_count:7d} | {mean_w:12.6f} | {expected_draws:16.1f} | {c_prob*100:7.2f}% | {ratio:16.2f}x")
    print()

audit_sampler("Official Baseline", base_sample_w)
audit_sampler("E3.4 (Both Human & Vehicle Capped)", e3_4_w)
audit_sampler("E3.5 Proposed (Human Uncapped, Vehicle Capped)", e3_5_w)
