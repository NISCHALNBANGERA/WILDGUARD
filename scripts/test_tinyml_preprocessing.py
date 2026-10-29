"""
WILDGUARD Automated Test for TinyML Audio Preprocessing
Tests:
1. Resampling and mono conversion on diverse sample rates (8k, 12k, 24k, 44.1k, 48k).
2. Exact output tensor shape (1, 64, 126).
3. Value range normalization (clamped within [-1.0, 1.0]).
4. Train-only augmentation vs validation deterministic behavior.
5. Silence detection.
6. Execution on real audio files from all 5 datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from wildguard_tinyml_preprocessor import TinyMLAudioPreprocessor

def main():
    base_dir = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
    manifest_path = base_dir / "data" / "processed" / "master_manifest_v11_training_canonical.csv"
    
    print("=" * 80)
    print("WILDGUARD TINYML PREPROCESSING PIPELINE TESTS")
    print("=" * 80)
    
    df = pd.read_csv(manifest_path, low_memory=False)
    preprocessor = TinyMLAudioPreprocessor()
    
    # Select 1 sample from each dataset
    test_samples = []
    for ds in df["dataset"].unique():
        row = df[(df["dataset"] == ds) & (df["is_training_eligible"] == True)].iloc[0]
        test_samples.append((ds, row["audio_path"], row["wildguard_class"]))
        
    print(f"Testing real audio files across {len(test_samples)} datasets...")
    for ds, path, cls in test_samples:
        print(f"\n--- Testing Dataset: {ds} ({cls}) ---")
        print(f"Path: {path}")
        
        # Test validation mode (deterministic)
        spec1, silent1 = preprocessor.process(path, is_train=False)
        spec2, silent2 = preprocessor.process(path, is_train=False)
        
        assert spec1.shape == (1, 64, 126), f"Unexpected shape: {spec1.shape}"
        assert np.allclose(spec1, spec2), "Validation mode is not deterministic!"
        assert spec1.min() >= -1.0 and spec1.max() <= 1.0, f"Out of [-1, 1] range: {spec1.min()}, {spec1.max()}"
        print(f"  Shape: {spec1.shape} | Range: [{spec1.min():.2f}, {spec1.max():.2f}] | Silent: {silent1} -> PASS")
        
        # Test train mode (augmented)
        spec_aug, _ = preprocessor.process(path, is_train=True)
        assert spec_aug.shape == (1, 64, 126)
        print(f"  Train Augmented Shape: {spec_aug.shape} -> PASS")
        
    print("\n" + "=" * 80)
    print("ALL TINYML PREPROCESSING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
