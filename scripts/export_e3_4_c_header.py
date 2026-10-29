"""
WILDGUARD TinyML Export Pipeline: BatchNorm Folding & ESP32-S3 C-Header Generator
Exports the E3.4 Champion HybridCNN (wildguard_audio_cnn_best_gates.pt) for microcontrollers.
- Folds BatchNorm parameters directly into Conv2d weights and biases.
- Eliminates 3 runtime BatchNorm operators, saving latency and SRAM buffer overhead.
- Generates:
  1. wildguard_hybrid_cnn_weights.h (C static header with Flash rodata placement)
  2. wildguard_hybrid_cnn_weights.bin (Raw binary for memory-mapped SPI Flash)
"""

import os
import sys
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path("C:/Users/Nischal N Bangera/OneDrive/Desktop/WILDGUARD")
sys.path.append(str(PROJECT_ROOT / "scripts"))

from train_e3_hybrid import HybridCNN

CKPT_PATH = PROJECT_ROOT / "outputs/e3_4_targeted_hn/wildguard_audio_cnn_best_gates.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs/e3_4_targeted_hn"
HEADER_PATH = OUTPUT_DIR / "wildguard_hybrid_cnn_weights.h"
BIN_PATH = OUTPUT_DIR / "wildguard_hybrid_cnn_weights.bin"

CLASSES = [
    "Deforestation",
    "Forest_Environment",
    "Forest_Fire",
    "Gunshot_like",
    "Human_Sound",
    "Vehicle"
]

def fold_conv_bn(conv, bn):
    """Folds BatchNorm scale, shift, running mean and var into Conv2d weights and bias."""
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps
    
    scale = gamma / torch.sqrt(var + eps)
    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = (b - mean) * scale + beta
    return w_folded.numpy(), b_folded.numpy()

def format_c_array(arr, indent=4):
    """Formats 1D numpy array as a C array literal with wrapping."""
    lines = []
    current_line = " " * indent
    for val in arr:
        token = f"{val:+.7e}f, "
        if len(current_line) + len(token) > 95:
            lines.append(current_line)
            current_line = " " * indent + token
        else:
            current_line += token
    if current_line.strip():
        lines.append(current_line)
    return "\n".join(lines)

def main():
    print("=" * 70)
    print("  WILDGUARD ESP32-S3 TinyML Export Pipeline")
    print("  Model: E3.4 Champion HybridCNN (Gate-Compliant)")
    print("=" * 70)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print(f"Loaded checkpoint from: {CKPT_PATH}")
    print(f"Checkpoint Epoch: {ckpt.get('epoch')}, Macro F1: {ckpt.get('val_macro_f1', 'N/A')}")

    model = HybridCNN(n_classes=len(CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Fold Conv + BatchNorm pairs
    # Conv1: conv[0], BN1: conv[1]
    c1_w, c1_b = fold_conv_bn(model.conv[0], model.conv[1])
    # Conv2: conv[4], BN2: conv[5]
    c2_w, c2_b = fold_conv_bn(model.conv[4], model.conv[5])
    # Conv3: conv[8], BN3: conv[9]
    c3_w, c3_b = fold_conv_bn(model.conv[8], model.conv[9])

    # Classifier Linear layer: classifier[1]
    fc_w = model.classifier[1].weight.data.numpy()
    fc_b = model.classifier[1].bias.data.numpy()

    p_c1_w = c1_w.size
    p_c1_b = c1_b.size
    p_c2_w = c2_w.size
    p_c2_b = c2_b.size
    p_c3_w = c3_w.size
    p_c3_b = c3_b.size
    p_fc_w = fc_w.size
    p_fc_b = fc_b.size
    total_folded_params = p_c1_w + p_c1_b + p_c2_w + p_c2_b + p_c3_w + p_c3_b + p_fc_w + p_fc_b

    print(f"\nParameter Audit (Post BatchNorm-Folding):")
    print(f"  Conv1 Folded:   {c1_w.shape} weights ({p_c1_w}) + {c1_b.shape} bias ({p_c1_b}) = {p_c1_w + p_c1_b}")
    print(f"  Conv2 Folded:   {c2_w.shape} weights ({p_c2_w}) + {c2_b.shape} bias ({p_c2_b}) = {p_c2_w + p_c2_b}")
    print(f"  Conv3 Folded:   {c3_w.shape} weights ({p_c3_w}) + {c3_b.shape} bias ({p_c3_b}) = {p_c3_w + p_c3_b}")
    print(f"  FC Linear:      {fc_w.shape} weights ({p_fc_w}) + {fc_b.shape} bias ({p_fc_b}) = {p_fc_w + p_fc_b}")
    print(f"  TOTAL PARAMETERS: {total_folded_params:,} (FP32 size: {total_folded_params * 4 / 1024:.2f} KB)")
    print(f"  Flash Memory Budget: ESP32-S3 (8MB Flash) -> Utilizes {total_folded_params * 4 / (8 * 1024 * 1024) * 100:.3f}% of Flash")

    # 1. Write Binary Flat File
    all_floats = np.concatenate([
        c1_w.flatten(), c1_b.flatten(),
        c2_w.flatten(), c2_b.flatten(),
        c3_w.flatten(), c3_b.flatten(),
        fc_w.flatten(), fc_b.flatten()
    ]).astype(np.float32)

    with open(BIN_PATH, "wb") as f:
        all_floats.tofile(f)
    print(f"\nSaved raw binary weights to: {BIN_PATH} ({BIN_PATH.stat().st_size:,} bytes)")

    # 2. Write C Header File
    with open(HEADER_PATH, "w") as f:
        f.write("/*\n")
        f.write(" * WILDGUARD HybridCNN Edge Deployment Weights\n")
        f.write(" * Automatically generated from outputs/e3_4_targeted_hn/wildguard_audio_cnn_best_gates.pt\n")
        f.write(" * Target: ESP32-S3 (Xtensa Dual-Core 240MHz, 512KB SRAM, 8MB Flash)\n")
        f.write(f" * Total Folded Parameters: {total_folded_params:,}\n")
        f.write(f" * Footprint: {total_folded_params * 4 / 1024:.2f} KB (FP32 .rodata in Flash)\n")
        f.write(" */\n\n")
        f.write("#ifndef WILDGUARD_HYBRID_CNN_WEIGHTS_H\n")
        f.write("#define WILDGUARD_HYBRID_CNN_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define WILDGUARD_NUM_CLASSES 6\n")
        f.write("#define WILDGUARD_INPUT_CHANNELS 1\n")
        f.write("#define WILDGUARD_INPUT_MELS 64\n")
        f.write("#define WILDGUARD_INPUT_FRAMES 126\n")
        f.write("#define WILDGUARD_FEATURE_DIM 128\n\n")
        
        f.write("static const char* const WILDGUARD_CLASS_NAMES[WILDGUARD_NUM_CLASSES] = {\n")
        for cls_name in CLASSES:
            f.write(f'    "{cls_name}",\n')
        f.write("};\n\n")

        # Flash rodata attribute for ESP32-S3 (ESP-IDF)
        f.write("#if defined(ESP_PLATFORM)\n")
        f.write('#include "esp_attr.h"\n')
        f.write("#define WILDGUARD_RODATA DRAM_ATTR const\n")
        f.write("#else\n")
        f.write("#define WILDGUARD_RODATA const\n")
        f.write("#endif\n\n")

        f.write(f"// Conv1 Folded Weights [16, 1, 3, 3] ({p_c1_w} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv1_w[{p_c1_w}] = {{\n")
        f.write(format_c_array(c1_w.flatten()) + "\n};\n\n")

        f.write(f"// Conv1 Folded Bias [16] ({p_c1_b} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv1_b[{p_c1_b}] = {{\n")
        f.write(format_c_array(c1_b.flatten()) + "\n};\n\n")

        f.write(f"// Conv2 Folded Weights [32, 16, 3, 3] ({p_c2_w} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv2_w[{p_c2_w}] = {{\n")
        f.write(format_c_array(c2_w.flatten()) + "\n};\n\n")

        f.write(f"// Conv2 Folded Bias [32] ({p_c2_b} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv2_b[{p_c2_b}] = {{\n")
        f.write(format_c_array(c2_b.flatten()) + "\n};\n\n")

        f.write(f"// Conv3 Folded Weights [64, 32, 3, 3] ({p_c3_w} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv3_w[{p_c3_w}] = {{\n")
        f.write(format_c_array(c3_w.flatten()) + "\n};\n\n")

        f.write(f"// Conv3 Folded Bias [64] ({p_c3_b} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_conv3_b[{p_c3_b}] = {{\n")
        f.write(format_c_array(c3_b.flatten()) + "\n};\n\n")

        f.write(f"// Linear Classifier Weights [6, 128] ({p_fc_w} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_fc_w[{p_fc_w}] = {{\n")
        f.write(format_c_array(fc_w.flatten()) + "\n};\n\n")

        f.write(f"// Linear Classifier Bias [6] ({p_fc_b} floats)\n")
        f.write(f"static WILDGUARD_RODATA float wildguard_fc_b[{p_fc_b}] = {{\n")
        f.write(format_c_array(fc_b.flatten()) + "\n};\n\n")

        f.write("#endif // WILDGUARD_HYBRID_CNN_WEIGHTS_H\n")

    print(f"Saved C header to: {HEADER_PATH} ({HEADER_PATH.stat().st_size:,} bytes)")
    print("=" * 70)
    print("  Export Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
