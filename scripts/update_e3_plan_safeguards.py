"""
WILDGUARD E3 — Update Experiment Plan & Config Proposal with Verified Safeguards
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
OUT_DIR = ROOT / "outputs/wildguard_e3_plan"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Update e3_config_proposal.json
    config = {
        "pipeline": "WILDGUARD E3 Controlled Improvement Pipeline",
        "date_updated": datetime.now().isoformat(),
        "baseline_champion": {
            "checkpoint": "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt",
            "status": "Official Leader (Preserved Untouched)",
            "validation_metrics": {
                "accuracy": 0.9084,
                "macro_f1": 0.7904,
                "weighted_f1": 0.9102,
                "gunshot_recall": 0.9565,
                "deforest_recall": 0.8880,
                "human_sound_precision": 0.4355,
                "vehicle_precision": 0.6170
            },
            "canonical_test_metrics": {
                "accuracy": 0.9222,
                "macro_f1": 0.8125,
                "weighted_f1": 0.9232,
                "gunshot_recall": 0.9806,
                "deforest_recall": 0.9038,
                "human_sound_precision": 0.5321,
                "vehicle_precision": 0.5000
            }
        },
        "target_hardware": {
            "platform": "ESP32-S3 (Xtensa Dual-Core LX7 @ 240 MHz)",
            "limits": {
                "internal_sram_kb": 512,
                "external_psram_mb": 8,
                "flash_mb": 8,
                "param_limit": 30000
            },
            "verified_architecture_footprint": {
                "SmallCNN": {
                    "parameters": 23910,
                    "flash_fp32_kb": 95.6,
                    "flash_int8_kb": 23.9,
                    "peak_activation_fp32_kb": 1592.0,
                    "peak_activation_int8_kb": 398.0,
                    "pooling_input_shape": [1, 64, 16, 99],
                    "total_macs": 62281856
                },
                "HybridCNN": {
                    "parameters": 24294,
                    "param_delta": 384,
                    "flash_fp32_kb": 97.2,
                    "flash_int8_kb": 24.3,
                    "peak_activation_fp32_kb": 1592.0,
                    "peak_activation_int8_kb": 398.0,
                    "pooling_input_shape": [1, 64, 16, 99],
                    "total_macs": 62282240,
                    "macs_delta": 384,
                    "macs_delta_pct": 0.0006
                }
            }
        },
        "validation_uncertainty_analysis": {
            "sample_support": {
                "Forest_Environment": 4507,
                "Deforestation": 3356,
                "Gunshot_like": 184,
                "Vehicle": 77,
                "Human_Sound": 64,
                "Forest_Fire": 25
            },
            "statistical_uncertainty_95ci": {
                "Deforestation_recall": "88.8% +/- 1.0%",
                "Gunshot_recall": "95.7% +/- 3.0%",
                "Vehicle_recall": "75.3% +/- 9.5%",
                "Human_Sound_recall": "84.4% +/- 8.9%",
                "Forest_Fire_recall": "96.0% +/- 9.4%"
            },
            "gate_interpretation_notes": "For small classes (Forest_Fire N=25, Human_Sound N=64, Vehicle N=77), single-sample flips cause 1.3% to 4.0% metric shifts. Regression checks consider confidence intervals rather than rigid point estimates."
        },
        "experiments": [
            {
                "experiment_id": "E3.1_HybridPooling",
                "name": "Hybrid Average/Max Temporal Pooling",
                "hypothesis": "Concatenating AdaptiveAvgPool2d and AdaptiveMaxPool2d preserves 20ms transient impact peaks (gunshot attack, axe strikes) without diluting into 4s background noise, while preserving continuous drone recognition.",
                "status": "Ready for Execution",
                "command": "python scripts/train_e3_hybrid.py --experiment e3_1_hybrid_pooling --epochs 10 --batch-size 32 --lr 0.001 --sampler capped_balanced",
                "output_dir": "outputs/e3_1_hybrid_pooling/",
                "expected_checkpoints": [
                    "outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt",
                    "outputs/e3_1_hybrid_pooling/checkpoint_latest.pt"
                ]
            },
            {
                "experiment_id": "E3.2_FocalLoss",
                "name": "Focal Loss & Logit Prior Calibration",
                "hypothesis": "Focal Loss (gamma=2.0) focuses training on hard boundary examples without shifting prior probabilities toward Human_Sound. (Treated as an empirical hypothesis requiring validation comparison).",
                "status": "Pending E3.1 Validation Result",
                "command": "python scripts/train_e3_hybrid.py --experiment e3_2_focal_loss --epochs 10 --batch-size 32 --loss focal --gamma 2.0 --sampler uniform",
                "output_dir": "outputs/e3_2_focal_loss/",
                "expected_checkpoints": [
                    "outputs/e3_2_focal_loss/wildguard_audio_cnn_best.pt"
                ]
            },
            {
                "experiment_id": "E3.3_MultiOffsetLogging",
                "name": "Multi-Offset Extraction for Manual Logging",
                "hypothesis": "Generating two overlapping 4.0s windows ([0-4s] and [1-5s]) from 5.0s FSC22 clips strictly within training partition doubles manual logging training density without cross-split leakage.",
                "status": "Pending E3.1/E3.2 Validation Result",
                "command": "python scripts/train_e3_hybrid.py --experiment e3_3_multi_offset --epochs 10 --batch-size 32 --manifest data/processed/master_manifest_v16_e3_multi_offset.csv",
                "output_dir": "outputs/e3_3_multi_offset/",
                "expected_checkpoints": [
                    "outputs/e3_3_multi_offset/wildguard_audio_cnn_best.pt"
                ]
            }
        ]
    }

    with open(OUT_DIR / "e3_config_proposal.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Updated e3_config_proposal.json")

    # 2. Update e3_experiment_plan.md
    plan_text = """# WILDGUARD E3 — Rigorous Controlled Improvement Plan

**Date:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """  
**Evaluation Status:** Preflight Verification Completed — Ready for Controlled Execution  
**Official Leader Checkpoint:** Baseline (`outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt`, Val Macro F1: **0.7904**, Test Macro F1: **0.8125**)  
**Integrity Guarantee:** Canonical Held-Out Test Set (**8,189 samples**) is **100% FROZEN** and untouched.  

---

## 1. Verified Architecture & Hardware Deployment Safeguards (ESP32-S3)

All architectural claims, tensor dimensions, and memory calculations were empirically verified using real PyTorch layer profiling on input tensor `(1, 1, 64, 398)`:

### A. Layer-by-Layer Dimension & Memory Profile
```
Input:               (1,  1, 64, 398) =   25,472 elements ( 101.9 KB FP32 /  25.5 KB INT8)
Layer 0  (Conv1):    (1, 16, 64, 398) =  407,552 elements (1592.0 KB FP32 / 398.0 KB INT8)
Layer 3  (MaxPool1): (1, 16, 32, 199) =  101,888 elements ( 398.0 KB FP32 /  99.5 KB INT8)
Layer 4  (Conv2):    (1, 32, 32, 199) =  203,776 elements ( 796.0 KB FP32 / 199.0 KB INT8)
Layer 7  (MaxPool2): (1, 32, 16,  99) =   50,688 elements ( 198.0 KB FP32 /  49.5 KB INT8)
Layer 8  (Conv3):    (1, 64, 16,  99) =  101,376 elements ( 396.0 KB FP32 /  99.0 KB INT8)
Pooling Input:       (1, 64, 16,  99) =  101,376 elements ( 396.0 KB FP32 /  99.0 KB INT8)
AvgPool Output:      (1, 64,  1,   1) =       64 elements (   0.25 KB FP32 /  0.06 KB INT8)
MaxPool Output:      (1, 64,  1,   1) =       64 elements (   0.25 KB FP32 /  0.06 KB INT8)
Concatenated Feature:(1, 128)         =      128 elements (   0.51 KB FP32 /  0.13 KB INT8)
Classifier Output:   (1, 6)           =        6 elements (   0.02 KB FP32 /  0.01 KB INT8)
```

### B. Hardware Comparison vs ESP32-S3 Limits
| Metric | Baseline SmallCNN | Proposed HybridCNN | ESP32-S3 Hardware Limit | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameter Count** | 23,910 | **24,294** (+384) | 30,000 parameter budget | **PASS (81.0% of limit)** |
| **Flash (FP32)** | 95.6 KB | **97.2 KB** (+1.6 KB) | 8 MB Octal SPI Flash | **PASS (<1.5% of Flash)** |
| **Flash (INT8 Quantized)** | 23.9 KB | **24.3 KB** (+384 B) | 8 MB Octal SPI Flash | **PASS (<0.5% of Flash)** |
| **Peak Activation Buffer (INT8)** | 398.0 KB (Layer 0) | **398.0 KB** (Layer 0, 0 change) | 512 KB Internal SRAM | **PASS (Fits in internal SRAM)** |
| **Peak Activation Buffer (FP32)** | 1,592 KB | **1,592 KB** (0 change) | 8 MB Octal SPI PSRAM | **PASS (Fits in PSRAM)** |
| **Total MACs / FLOPs per Window** | 62,281,856 | **62,282,240** (+384 MACs) | 240 MHz Dual-Core LX7 | **PASS (+0.0006% compute)** |
| **Transient Attenuation Factor** | **33.4x Dilution** (Peak 5.0 -> 0.15) | **100% Preserved** (Peak 5.0 -> 5.0) | N/A | **Acoustic Transients Preserved** |

---

## 2. Multi-Offset Window Generation & Split Isolation Protocol

To address the manual logging data shortage without downloading external datasets:
1. **Multi-Offset Strategy:** All FSC22 manual logging clips are 5.0 seconds long (`Axe`, `WoodChop`, `TreeFalling`, `Handsaw`). We generate two 4.0-second windows per recording:
   - Window A: `[0.0s to 4.0s]` (initial strike / onset)
   - Window B: `[1.0s to 5.0s]` (secondary resonance / fracture tail)
2. **Absolute Split Isolation Rule:**
   - Multiple windows from one source recording MUST NEVER cross splits:
     $$\text{SliceSplit}(R_i, \text{offset}) \equiv \text{ParentSplit}(R_i)$$
   - If parent recording $R_i \in \text{Train}$, both Window A and Window B are assigned strictly to **Train**.
   - If parent recording $R_j \in \text{Validation}$, both Window A and Window B are assigned strictly to **Validation**.
   - **Canonical Test Set Isolation:** Strictly **0 new slices** are added to the test partition.
3. **Verified Parent Group Isolation:**
   - Audited all 42,049 parent groups in `master_manifest_v15_e2_augmented.csv`: exactly **0 parent groups cross train/validation/test boundaries**.
4. **Sample Weighting Protocol:**
   - To prevent multiple offsets from artificially inflating the loss contribution of a single parent recording, each offset window is assigned weight $w / 2$ where $w$ is the base sample weight.

---

## 3. Validation Computability, Regression Gates & Statistical Uncertainty

All regression checks are computable directly from the **8,213-sample validation split**. Evaluating Baseline and E2 on this split confirmed that validation performance directly predicts test performance:

### A. Validation Support & Statistical Confidence Intervals (95% Wilson CI)
| Class | Val Samples ($N$) | Proportion | Baseline Val Recall | 95% Confidence Interval | Sample Flip Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deforestation** | 3,356 | 40.86% | 88.80% | [87.7%, 89.8%] | 1 sample = 0.03% |
| **Forest_Environment** | 4,507 | 54.88% | 92.50% | [91.7%, 93.2%] | 1 sample = 0.02% |
| **Gunshot_like** | 184 | 2.24% | 95.65% | [91.7%, 97.8%] | 1 sample = 0.54% |
| **Vehicle** | 77 | 0.94% | 75.32% | [64.7%, 83.6%] | 1 sample = 1.30% |
| **Human_Sound** | 64 | 0.78% | 84.38% | [73.6%, 91.3%] | 1 sample = 1.56% |
| **Forest_Fire** | 25 | 0.30% | 96.00% | [80.5%, 99.3%] | 1 sample = 4.00% |

### B. Mandatory Validation Regression Gates (Disqualification Rules)
A candidate model is **disqualified immediately** if it fails ANY of the following on the validation set:
1. **Validation Gunshot Recall < 94.0%** (Allowing for 184-sample binomial uncertainty, minimum 173 / 184 correct).
   * *Audit Evidence:* Baseline achieved 95.65% (176/184); E2 dropped to 86.96% (160/184) and was disqualified.
2. **Validation Deforestation Recall < 87.5%** (Minimum 2,936 / 3,356 correct).
   * *Audit Evidence:* Baseline achieved 88.80%; E2 dropped to 84.62% and was disqualified.
3. **Validation Human_Sound Precision < 40.0%** (Prevention of severe false alarm flood).
   * *Audit Evidence:* Baseline achieved 43.55%; E2 collapsed to 27.48% and was disqualified.
4. **Validation Vehicle Precision < 55.0%** (Preservation of ground combustion engine recognition).
   * *Audit Evidence:* Baseline achieved 61.70%; E2 achieved 73.86%.

---

## 4. Methodological Safeguard: Hypotheses vs Assumed Improvements

1. **Hybrid Pooling is a Hypothesis:**
   - While max-pooling preserves 20ms transients, it could theoretically increase sensitivity to microphone clicks or static bursts. Validation evaluation will verify whether Gunshot precision remains high.
2. **Focal Loss is a Hypothesis:**
   - Down-weighting easy examples ($\gamma = 2.0$) helps hard boundaries, but could destabilize learning on massive ambient classes (RFCx). It must strictly beat CrossEntropy on the validation split before adoption.
3. **Probability Thresholding is a Calibration Hypothesis:**
   - Adjusting thresholds ($P > 0.35$ or $P > 0.65$) directly trades off precision for recall. Thresholds will be tuned on validation data only and tested for side-effects before deployment.

---

## 5. Controlled Experiment Execution Plan

The execution follows a strict sequential ladder where each experiment isolates exactly one variable:

### Experiment E3.1: Hybrid Average/Max Temporal Pooling
* **Objective:** Test if concatenating Average and Max pooling recovers gunshots and axe strikes while capping Human_Sound sampler weight to 1.0x.
* **Exact Command:**
  ```powershell
  python scripts/train_e3_hybrid.py --experiment e3_1_hybrid_pooling --epochs 10 --batch-size 32 --lr 0.001 --sampler capped_balanced
  ```
* **Expected Output Artifacts:**
  - `outputs/e3_1_hybrid_pooling/wildguard_audio_cnn_best.pt` (best validation checkpoint)
  - `outputs/e3_1_hybrid_pooling/checkpoint_latest.pt` (epoch 10 resume checkpoint)
  - `outputs/e3_1_hybrid_pooling/training_history.json`
* **Success Gate:** Validation Macro F1 > 0.7904 with Gunshot Recall $\ge 94.0\%$ and Deforestation Recall $\ge 87.5\%$.

### Experiment E3.2: Focal Loss & Logit Prior Calibration (Conditional on E3.1)
* **Objective:** Test if Focal Loss ($\gamma = 2.0$) with uniform sampling beats weighted cross-entropy.
* **Exact Command:**
  ```powershell
  python scripts/train_e3_hybrid.py --experiment e3_2_focal_loss --epochs 10 --batch-size 32 --loss focal --gamma 2.0 --sampler uniform
  ```
* **Expected Output Artifacts:**
  - `outputs/e3_2_focal_loss/wildguard_audio_cnn_best.pt`
  - `outputs/e3_2_focal_loss/training_history.json`
* **Success Gate:** Validation Macro F1 strictly exceeds E3.1 and Baseline without regression.

### Experiment E3.3: Multi-Offset Manual Logging Enhancement (Conditional on E3.1/E3.2)
* **Objective:** Test if 2x multi-offset windowing (662 training samples) from 5.0s FSC22 clips improves validation manual logging recall.
* **Exact Command:**
  ```powershell
  python scripts/train_e3_hybrid.py --experiment e3_3_multi_offset --epochs 10 --batch-size 32 --manifest data/processed/master_manifest_v16_e3_multi_offset.csv
  ```
* **Expected Output Artifacts:**
  - `outputs/e3_3_multi_offset/wildguard_audio_cnn_best.pt`
  - `outputs/e3_3_multi_offset/training_history.json`
* **Success Gate:** Validation Manual Logging Recall $\ge 35\%$ while maintaining all global regression gates.

---

## 6. Artifact Inventory

- **Updated Experiment Plan:** [`outputs/wildguard_e3_plan/e3_experiment_plan.md`](file:///C:/Users/Nischal%20N%20Bangera/OneDrive/Desktop/WILDGUARD/outputs/wildguard_e3_plan/e3_experiment_plan.md)
- **Updated Config Proposal:** [`outputs/wildguard_e3_plan/e3_config_proposal.json`](file:///C:/Users/Nischal%20N%20Bangera/OneDrive/Desktop/WILDGUARD/outputs/wildguard_e3_plan/e3_config_proposal.json)
- **Risk Register:** [`outputs/wildguard_e3_plan/e3_risk_register.csv`](file:///C:/Users/Nischal%20N%20Bangera/OneDrive/Desktop/WILDGUARD/outputs/wildguard_e3_plan/e3_risk_register.csv)
"""

    with open(OUT_DIR / "e3_experiment_plan.md", "w", encoding="utf-8") as f:
        f.write(plan_text)
    print("Updated e3_experiment_plan.md")

if __name__ == "__main__":
    main()
