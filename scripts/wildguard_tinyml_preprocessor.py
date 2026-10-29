"""
WILDGUARD TinyML Audio Preprocessing & Feature Extraction Pipeline
Phase 5 - Prepares standardized 16kHz mono audio and log-Mel spectrograms for TinyML (ESP32-S3).
Features:
- Mono conversion & 16 kHz resampling on-the-fly (keeps original audio untouched)
- Fixed 4.0-second windowing (64,000 samples)
- Center-padding for short clips (< 4s), center-cropping or chunking for long clips (> 4s)
- Log-Mel spectrogram extraction (64 mel bins, n_fft=1024, hop_length=512)
- TinyML-compliant output shapes: (1, 64, 126) suitable for Micro-CNNs
- Train-only augmentation (time shift, additive noise, SpecAugment)
- Energy threshold silence detection
"""
import os
import math
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

TARGET_SR = 16000
WINDOW_DURATION = 4.0
WINDOW_SAMPLES = int(TARGET_SR * WINDOW_DURATION) # 64,000 samples

# Log-Mel Spectrogram parameters optimized for TinyML / ESP32-S3 memory constraints
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
F_MIN = 50.0
F_MAX = 8000.0

class TinyMLAudioPreprocessor:
    def __init__(self, target_sr=TARGET_SR, duration=WINDOW_DURATION, n_mels=N_MELS):
        self.target_sr = target_sr
        self.duration = duration
        self.n_samples = int(target_sr * duration)
        self.n_mels = n_mels
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        self.f_min = F_MIN
        self.f_max = F_MAX
        
    def load_audio(self, audio_path):
        """Loads and converts any audio to mono, 16 kHz."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return y, sr

    def pad_or_crop(self, y):
        """Standardizes audio to exact window_samples (64,000 samples)."""
        length = len(y)
        if length == self.n_samples:
            return y
        elif length < self.n_samples:
            # Short clip (e.g. RFCx 3s or gunshots): pad center
            pad_left = (self.n_samples - length) // 2
            pad_right = self.n_samples - length - pad_left
            return np.pad(y, (pad_left, pad_right), mode='constant')
        else:
            # Long clip (e.g. FSC22 5s or ESC-50 5s): center crop
            start = (length - self.n_samples) // 2
            return y[start:start + self.n_samples]

    def normalize_waveform(self, y, eps=1e-6):
        """Peak amplitude normalization."""
        peak = np.max(np.abs(y))
        if peak > eps:
            return y / peak
        return y

    def is_silent(self, y, energy_thresh=1e-4):
        """Detects whether clip has negligible acoustic energy."""
        energy = np.mean(y ** 2)
        return energy < energy_thresh

    def apply_augmentation(self, y, is_train=True):
        """Applies train-only data augmentation."""
        if not is_train:
            return y
            
        # 1. Random time shift (+/- 0.5 sec)
        if random.random() < 0.5:
            shift = int(random.uniform(-0.5, 0.5) * self.target_sr)
            y = np.roll(y, shift)
            
        # 2. Add subtle Gaussian noise (SNR ~20-30dB)
        if random.random() < 0.3:
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise
            
        return y

    def extract_log_mel_spectrogram(self, y):
        """Computes log-Mel spectrogram matching ESP32-S3 tensor shapes."""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [-1.0, 1.0] range for fixed-point quantization / INT8 inference
        log_mel_norm = (log_mel + 40.0) / 40.0
        log_mel_norm = np.clip(log_mel_norm, -1.0, 1.0)
        
        # Add channel dimension: (1, n_mels, time_steps)
        return np.expand_dims(log_mel_norm.astype(np.float32), axis=0)

    def process(self, audio_path, is_train=False):
        """Full end-to-end processing pipeline for a single file."""
        y, _ = self.load_audio(audio_path)
        y = self.pad_or_crop(y)
        y = self.normalize_waveform(y)
        
        if is_train:
            y = self.apply_augmentation(y, is_train=True)
            y = self.normalize_waveform(y)
            
        silent = self.is_silent(y)
        spec = self.extract_log_mel_spectrogram(y)
        return spec, silent
