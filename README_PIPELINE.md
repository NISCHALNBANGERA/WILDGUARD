# WILDGUARD — Autonomous Dataset Engineering & TinyML Pipeline

This repository hosts the complete, reproducible, leakage-safe dataset preparation workflow for **WILDGUARD**, an edge-AI forest acoustic intelligence system designed for embedded deployment on the **ESP32-S3** with LoRa alerting.

---

## 1. Target Audio Classes

The acoustic classifier distinguishes the following 7 core forest sound categories:

1. **Deforestation**: Chainsaw, axe, handsaw, wood chopping, tree falling (`RFCx`, `FSC22`, `ESC-50`).
2. **Forest_Environment**: Ordinary background ambience, birds, frogs, insects, crickets, rain, wind, water (`RFCx`, `ForestSoundDataset`, `FSC22`, `ESC-50`).
3. **Gunshot_like**: Firearm discharge and impulsive poaching sounds (`GunshotDataset` edge-collected recordings, `FSC22`).
4. **Human_Sound**: Conversational speech, shouting, footsteps, clapping, breathing (`FSC22`, `ESC-50`).
5. **Vehicle**: Ground vehicles including engines, horns, sirens (`FSC22`, `ESC-50`).
6. **Wild_Animal**: Key wildlife vocalizations including lion, wolf howl (`FSC22`).
7. **Forest_Fire**: Auxiliary fire crackling audio (`FSC22`, `ForestSoundDataset`, `ESC-50`; auxiliary target since fire detection is primarily environmental sensor-based).

> **Important Safety & Quarantine Rules:**
> - Domestic animals are strictly excluded from `Wild_Animal`.
> - Ambiguous sounds (e.g. 608 unlabeled poaching folder clips and 75 generator clips) are strictly quarantined as `UNASSIGNED` rather than guessed.
> - Exact duplicate copies across datasets are strictly quarantined from training.

---

## 2. Dataset Provenance & Physical Audio Inventory

All **58,291** audio files physically exist on disk and have passed 100% readability verification:

| Dataset | Verified Audio Files | Format & Channels | Sample Rates (Hz) | Primary Roles |
| :--- | :--- | :--- | :--- | :--- |
| **RFCx FrugalAI** | 50,392 | WAV (1 channel) | 12000, 24000 | Deforestation, Forest_Environment |
| **ForestSoundDataset** | 2,761 | WAV (1, 2, 8 ch) | 8000–192000 | Forest_Environment, Forest_Fire |
| **GunshotDataset** | 2,148 | WAV (1 channel) | 44100 | Gunshot_like (outdoor edge-collected) |
| **FSC22** | 1,950 | WAV (1, 2, 8 ch) | 8000–192000 | Ground-truth core backbone across all classes |
| **ESC-50** | 1,040 | WAV (1 channel) | 44100 | Supplementary environmental, human, vehicle |
| **Total** | **58,291** | **All Verified OK** | **Fully Audited** | **100% Physical Integrity** |

---

## 3. Versioned Manifests & Deliverables

All outputs are preserved in `data/processed/` without modifying or overwriting any raw audio:

1. **Validated Master Manifest**: `data/processed/master_manifest_v11_training_canonical.csv` (58,291 rows).
2. **Canonical Training Candidates**: 55,656 training-eligible canonical samples (2,027 duplicate copies and 683 unassigned rows safely quarantined).
3. **Group-Safe Split Plan**: `data/processed/master_manifest_v10_split_plan.csv` / `split_components_v1.csv` (70.8% Train, 14.6% Val, 14.7% Test).
4. **Exact Duplicate Audit**: `data/processed/duplicate_groups_v2.csv` (2,026 duplicate groups reconstructed).
5. **Label Conflict Audit**: `data/processed/label_conflicts_v2.csv` (690 conflict groups documented).
6. **Missing/Corrupted Audio Report**: `data/processed/audio_validation_report_v1.csv` & `missing_corrupted_audio_summary_v1.txt` (0 missing, 0 corrupted).
7. **Split & Leakage Audit**: `data/processed/split_leakage_summary_20261018_121414.txt` & `split_plan_summary_v1.txt` (0 duplicate group leakage, 0 source group leakage).
8. **Class Distribution Report**: `data/processed/class_distribution_report_v1.csv` & `class_distribution_summary_v1.txt`.
9. **Automated Manifest Test Suite**: `scripts/test_manifest_validation.py` (10/10 automated assertions passed).
10. **TinyML Audio Preprocessor**: `scripts/wildguard_tinyml_preprocessor.py` & `scripts/test_tinyml_preprocessing.py`.
11. **Pipeline Orchestrator**: `scripts/run_wildguard_pipeline.py`.

---

## 4. Leakage-Safe Splitting Architecture

- **Connected Component Grouping**: Audio clips connected by dataset `source_group_id`, RFCx source UUID, or cross-dataset SHA-256 duplicate groups are merged into unified connected components.
- **Strict Disjointness**: All clips in a component are guaranteed to reside in the same split (`train`, `validation`, or `test`). Zero chunk, duplicate, or recording leakage exists across splits.

---

## 5. TinyML Embedded Specifications (ESP32-S3)

- **Target Audio Format**: Single-channel mono, 16,000 Hz.
- **Window Size**: 4.0 seconds (64,000 samples).
  - Short clips (< 4.0s) are center-padded with zeros.
  - Long clips (> 4.0s) are center-cropped or chunked.
- **Log-Mel Spectrogram**:
  - `N_MELS`: 64
  - `N_FFT`: 1024 (64 ms window)
  - `HOP_LENGTH`: 512 (32 ms hop)
  - `Tensor Output Shape`: `(1, 64, 126)`
- **Dynamic Range**: Normalized and clamped to `[-1.0, 1.0]` for INT8 CMSIS-NN quantization.
- **Train-Only Augmentation**: Time-shift (+/- 0.5s) and additive acoustic noise applied strictly to the `train` split.

---

## 6. One-Command Execution Instructions

To execute the entire end-to-end audit, validation, manifest generation, and TinyML test suite:

```bash
python scripts/run_wildguard_pipeline.py
```

To run individual modules:
```bash
# Validate physical audio integrity:
python scripts/validate_audio_integrity.py

# Generate canonical manifest & class distributions:
python scripts/create_canonical_training_manifest.py
python scripts/generate_class_distribution_report.py

# Run automated manifest validation assertions:
python scripts/test_manifest_validation.py

# Run TinyML preprocessing tests:
python scripts/test_tinyml_preprocessing.py
```
