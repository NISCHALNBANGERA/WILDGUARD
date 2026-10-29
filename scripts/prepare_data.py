import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

TARGET_SR = 16000
CHUNK_DURATION = 4.0 # seconds
OVERLAP = 2.0 # seconds
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)
STEP_SAMPLES = int(TARGET_SR * (CHUNK_DURATION - OVERLAP))

CLASSES = [
    "Forest Environment",
    "Chainsaw",
    "Gunshot",
    "Fire/Burning",
    "Human Speech",
    "Vehicle",
    "Other/Unknown"
]

def load_and_resample(file_path):
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def split_into_chunks(y):
    chunks = []
    if len(y) < CHUNK_SAMPLES:
        # Pad if smaller than 4 seconds
        pad_len = CHUNK_SAMPLES - len(y)
        y_padded = librosa.util.pad_center(data=y, size=CHUNK_SAMPLES)
        chunks.append(y_padded)
    else:
        for i in range(0, len(y) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
            chunk = y[i:i + CHUNK_SAMPLES]
            chunks.append(chunk)
    return chunks

def process_dataset(raw_files, class_map, dataset_name, out_dir):
    """
    raw_files: list of dicts with 'path', 'label', 'original_id'
    class_map: dict mapping raw labels to WILDGUARD classes
    """
    processed_records = []
    
    # Group by original_id for train/val/test split
    unique_ids = list(set([x['original_id'] for x in raw_files]))
    
    if len(unique_ids) == 0:
        return []
        
    # Attempt stratify by getting a majority label for each id, but simple split is safer for variable lengths
    try:
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    except ValueError:
        # If dataset is too small
        train_ids = unique_ids
        val_ids = []
        test_ids = []
        
    id_to_split = {}
    for i in train_ids: id_to_split[i] = 'train'
    for i in val_ids: id_to_split[i] = 'val'
    for i in test_ids: id_to_split[i] = 'test'
    
    for item in tqdm(raw_files, desc=f"Processing {dataset_name}"):
        path = item['path']
        raw_label = item['label']
        orig_id = item['original_id']
        
        target_class = class_map.get(raw_label, "Other/Unknown")
        if target_class not in CLASSES:
            target_class = "Other/Unknown"
            
        split = id_to_split[orig_id]
        
        y = load_and_resample(path)
        if y is None:
            continue
            
        chunks = split_into_chunks(y)
        for idx, chunk in enumerate(chunks):
            chunk_filename = f"{dataset_name}_{orig_id}_{idx}.wav"
            chunk_path = out_dir / split / chunk_filename
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(chunk_path, chunk, TARGET_SR)
            
            processed_records.append({
                'filename': chunk_filename,
                'path': str(chunk_path.resolve()),
                'dataset': dataset_name,
                'original_id': orig_id,
                'class': target_class,
                'split': split,
                'duration': len(chunk)/TARGET_SR,
                'sample_rate': TARGET_SR
            })
            
    return processed_records

def prep_esc50(raw_dir):
    # ESC-50 classes to map
    # Crickets(17), Chirping birds(14), Wind(16), Rain(10) -> Forest Environment
    # Crackling fire(12) -> Fire/Burning
    # Chainsaw(41) -> Chainsaw
    # Helicopter(40), Airplane(47), Train(46), etc -> Other/Unknown
    esc50_path = raw_dir / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv"
    if not esc50_path.exists():
        return []
    
    df = pd.read_csv(esc50_path)
    raw_files = []
    class_map = {
        'crickets': 'Forest Environment',
        'chirping_birds': 'Forest Environment',
        'wind': 'Forest Environment',
        'rain': 'Forest Environment',
        'crackling_fire': 'Fire/Burning',
        'chainsaw': 'Chainsaw'
    }
    
    # Rest to Other/Unknown
    for _, row in df.iterrows():
        raw_label = row['category']
        path = raw_dir / "ESC-50" / "ESC-50-master" / "audio" / row['filename']
        # orig_id is the source recording. ESC-50 filenames format: [fold]-[clip_id]-[src]-[category].wav
        # The src is the 3rd part
        parts = row['filename'].split('-')
        orig_id = parts[2] if len(parts) >= 3 else row['filename']
        
        raw_files.append({
            'path': path,
            'label': raw_label,
            'original_id': f"esc50_{orig_id}"
        })
    
    return raw_files, class_map

def prep_rfcx(raw_dir):
    audio_dir = raw_dir / "RFCx_FrugalAI" / "audio"
    if not audio_dir.exists():
        return []
        
    raw_files = []
    class_map = {
        'chainsaw': 'Chainsaw',
        'environment': 'Forest Environment'
    }
    
    for f in audio_dir.glob("*.wav"):
        if "pos" in f.name:
            label = "chainsaw"
        else:
            label = "environment"
            
        orig_id = f.stem
        raw_files.append({
            'path': f,
            'label': label,
            'original_id': f"rfcx_{orig_id}"
        })
        
    return raw_files, class_map

def prep_wildfire(raw_dir):
    audio_dir = raw_dir / "ForestWildFire"
    if not audio_dir.exists():
        return []
        
    raw_files = []
    class_map = {
        'fire': 'Fire/Burning'
    }
    
    for f in audio_dir.rglob("*.wav"):
        orig_id = f.stem
        raw_files.append({
            'path': f,
            'label': 'fire',
            'original_id': f"wildfire_{orig_id}"
        })
        
    return raw_files, class_map

def prep_fsc22(raw_dir):
    audio_dir = raw_dir / "FSC22" / "Audio_data" / "Audio_data"
    meta_path = raw_dir / "FSC22" / "Metadata" / "meta_data.csv"
    
    if not audio_dir.exists() or not meta_path.exists():
        return []
        
    df = pd.read_csv(meta_path)
    raw_files = []
    
    # FSC22 mapping
    # Mechanical sounds, vehicles, environment, etc.
    class_map = {
        'chainsaw': 'Chainsaw',
        'fire': 'Fire/Burning',
        'gun_shot': 'Gunshot',
        'general_speaking': 'Human Speech',
        'vehicle_engine': 'Vehicle',
        'birds': 'Forest Environment',
        'insects': 'Forest Environment',
        'rain': 'Forest Environment',
        'wind': 'Forest Environment',
        'thunderstorm': 'Forest Environment',
        'frogs': 'Forest Environment'
    }
    # Others to Other/Unknown
    
    for _, row in df.iterrows():
        fname = row['FileName']
        label = row['Class']
        path = audio_dir / fname
        orig_id = fname.split('_')[0] if '_' in fname else fname
        
        if path.exists():
            raw_files.append({
                'path': path,
                'label': label.lower(),
                'original_id': f"fsc22_{orig_id}"
            })
            
    return raw_files, class_map

def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    out_dir = base_dir / 'data' / 'processed'
    meta_dir = base_dir / 'data' / 'metadata'
    
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    print("Preparing ESC-50...")
    res = prep_esc50(raw_dir)
    if res:
        records = process_dataset(res[0], res[1], "ESC-50", out_dir)
        all_records.extend(records)
        
    print("Preparing RFCx FrugalAI...")
    res = prep_rfcx(raw_dir)
    if res:
        records = process_dataset(res[0], res[1], "RFCx", out_dir)
        all_records.extend(records)
        
    print("Preparing Kaggle WildFire...")
    res = prep_wildfire(raw_dir)
    if res:
        records = process_dataset(res[0], res[1], "WildFire", out_dir)
        all_records.extend(records)
        
    print("Preparing FSC22...")
    res = prep_fsc22(raw_dir)
    if res:
        records = process_dataset(res[0], res[1], "FSC22", out_dir)
        all_records.extend(records)
        
    df_meta = pd.DataFrame(all_records)
    if len(df_meta) > 0:
        csv_path = meta_dir / 'dataset_metadata.csv'
        df_meta.to_csv(csv_path, index=False)
        print(f"\nMetadata saved to {csv_path}")
        print("\nClass Distribution:")
        print(df_meta['class'].value_counts())
        print("\nSplit Distribution:")
        print(df_meta['split'].value_counts())
    else:
        print("No data processed. Ensure downloads completed successfully.")

if __name__ == "__main__":
    main()
