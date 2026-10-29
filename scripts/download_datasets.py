import os
import zipfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[{dest_path}] already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_esc50(raw_dir):
    esc50_dir = raw_dir / 'ESC-50'
    esc50_dir.mkdir(parents=True, exist_ok=True)
    zip_path = esc50_dir / 'ESC-50-master.zip'
    
    if not os.path.exists(esc50_dir / 'ESC-50-master'):
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        download_file(url, zip_path)
        if zip_path.exists():
            extract_zip(zip_path, esc50_dir)
            os.remove(zip_path)

def download_kaggle_datasets(raw_dir):
    try:
        fsc22_dir = raw_dir / 'FSC22'
        if not fsc22_dir.exists() or len(list(fsc22_dir.glob("*"))) == 0:
            print("Downloading FSC22 from Kaggle...")
            fsc22_dir.mkdir(parents=True, exist_ok=True)
            os.system(f"kaggle datasets download -d irmiot22/fsc22-dataset -p \"{fsc22_dir}\" --unzip")
            print(f"FSC22 downloaded to {fsc22_dir}")
        else:
            print("FSC22 already exists.")
            
        wildfire_dir = raw_dir / 'ForestWildFire'
        if not wildfire_dir.exists() or len(list(wildfire_dir.glob("*"))) == 0:
            print("Downloading Forest Wild Fire from Kaggle...")
            wildfire_dir.mkdir(parents=True, exist_ok=True)
            os.system(f"kaggle datasets download -d forestprotection/forest-wild-fire-sound-dataset -p \"{wildfire_dir}\" --unzip")
            print(f"Forest Wild Fire downloaded to {wildfire_dir}")
        else:
            print("Forest Wild Fire already exists.")
            
    except Exception as e:
        print(f"Failed to download Kaggle datasets: {e}")

def download_huggingface_datasets(raw_dir):
    try:
        from datasets import load_dataset
        import soundfile as sf
        
        rfcx_dir = raw_dir / 'RFCx_FrugalAI'
        rfcx_dir.mkdir(parents=True, exist_ok=True)
        
        # We just download a small sample for demonstration if it's too large, but let's try the whole train split
        # We will save them as wav files for consistent processing later
        if not (rfcx_dir / 'audio').exists():
            print("Downloading RFCx FrugalAI from HuggingFace...")
            (rfcx_dir / 'audio').mkdir()
            ds = load_dataset("rfcx/frugalai", split="train", streaming=True)
            
            # Download first 100 positive and 100 negative as a supplementary subset to save time/space
            pos_count = 0
            neg_count = 0
            max_samples = 200 # Adjust this if you want the full dataset
            
            print(f"Downloading a subset of RFCx FrugalAI ({max_samples*2} samples)...")
            for item in tqdm(ds):
                label = item['label']
                if label == 0 and pos_count < max_samples:
                    # Chainsaw (0)
                    audio_array = item['audio']['array']
                    sr = item['audio']['sampling_rate']
                    sf.write(rfcx_dir / 'audio' / f"rfcx_pos_{pos_count}.wav", audio_array, sr)
                    pos_count += 1
                elif label == 1 and neg_count < max_samples:
                    # Environment (1)
                    audio_array = item['audio']['array']
                    sr = item['audio']['sampling_rate']
                    sf.write(rfcx_dir / 'audio' / f"rfcx_neg_{neg_count}.wav", audio_array, sr)
                    neg_count += 1
                    
                if pos_count >= max_samples and neg_count >= max_samples:
                    break
            print("RFCx FrugalAI subset downloaded.")
        else:
            print("RFCx FrugalAI already exists.")
            
    except Exception as e:
        print(f"Failed to download HuggingFace dataset: {e}")

def main():
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting dataset downloads...")
    download_esc50(raw_dir)
    download_kaggle_datasets(raw_dir)
    download_huggingface_datasets(raw_dir)
    print("Downloads complete. Note: Zenodo datasets (DataSEC, TUT, 13893977) require manual or separate API downloads due to Zenodo rate limits and large sizes. Please download them manually if needed, or we can rely on FSC22, RFCx, Wildfire, and ESC50 for this execution.")

if __name__ == "__main__":
    main()
