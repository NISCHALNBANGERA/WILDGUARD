"""
Automated End-to-End Test for WILDGUARD Streamlit Application Logic
Tests:
1. Checkpoint file presence and format
2. Model loading and weights integrity
3. 6-class mapping and order
4. Preprocessing of all 7 demo sample files (6 classes + 1 silent)
5. Model forward pass, output shape (1, 6), probability bounds [0, 1]
6. Silence detection flag verification
7. Corrupted audio graceful exception handling
"""
from pathlib import Path
import sys
import io
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app import load_model, preprocess_audio_bytes, CLASSES, CKPT_PATH, SAMPLES_DIR

def run_tests():
    print("=" * 70)
    print("WILDGUARD DEMO APPLICATION END-TO-END AUTOMATED TEST")
    print("=" * 70)

    # Test 1: Checkpoint exists and loads
    print("\n[TEST 1] Loading model checkpoint...")
    assert CKPT_PATH.is_file(), f"Checkpoint missing: {CKPT_PATH}"
    model, ckpt = load_model()
    assert ckpt.get("classes") == CLASSES, f"Classes mismatch: {ckpt.get('classes')} vs {CLASSES}"
    print(f"  PASS: Loaded model with {len(CLASSES)} classes: {CLASSES}")

    # Test 2: Verify all 7 demo sample files
    sample_files = sorted(list(SAMPLES_DIR.glob("*.wav")))
    print(f"\n[TEST 2] Verifying preset demo samples ({len(sample_files)} files found)...")
    assert len(sample_files) >= 7, f"Expected at least 7 demo samples, found {len(sample_files)}"

    for sf in sample_files:
        audio_bytes = sf.read_bytes()
        tensor, audio_window, logmel, raw_dur, peak, rms, is_silent = preprocess_audio_bytes(audio_bytes, sf.name)
        assert tensor.shape == (1, 1, 64, 398), f"Unexpected tensor shape: {tensor.shape}"
        assert not np.isnan(logmel).any(), f"NaN in logmel for {sf.name}"

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        assert probs.shape == (6,), f"Unexpected probs shape: {probs.shape}"
        assert np.isclose(probs.sum(), 1.0, atol=1e-4), f"Probabilities do not sum to 1: {probs.sum()}"
        top_idx = int(np.argmax(probs))
        top_cls = CLASSES[top_idx]
        top_p = probs[top_idx]

        silent_str = " [SILENT DETECTED]" if is_silent else ""
        print(f"  PASS: {sf.name:<32} -> Top: {top_cls:<20} ({top_p*100:5.1f}%) | Dur: {raw_dur:.1f}s{silent_str}")

    # Test 3: Verify silence detection
    print("\n[TEST 3] Testing silence detection...")
    silent_path = SAMPLES_DIR / "silent_sample.wav"
    _, _, _, _, peak, rms, is_silent = preprocess_audio_bytes(silent_path.read_bytes(), "silent_sample.wav")
    assert is_silent, f"Silent file was not flagged as silent! peak={peak}, rms={rms}"
    print(f"  PASS: Silence correctly flagged (peak={peak:.6f}, rms={rms:.6f})")

    # Test 4: Verify corrupted audio handling
    print("\n[TEST 4] Testing invalid/corrupted audio handling...")
    corrupted_bytes = b"NOT_A_VALID_WAV_HEADER_CORRUPTED_BYTES"
    caught = False
    try:
        preprocess_audio_bytes(corrupted_bytes, "corrupted.wav")
    except Exception as e:
        caught = True
        print(f"  PASS: Correctly raised exception on invalid audio: {type(e).__name__}")
    assert caught, "Failed to catch corrupted audio exception!"

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
