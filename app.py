"""
WILDGUARD Acoustic Intelligence System — Local Demonstration
Model: SmallCNN 6-Class Audio Classifier
Checkpoint: outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt
"""
from pathlib import Path
import io
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import streamlit as st

# Root & paths
ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "outputs/audio_hard_negative_experiment_rerun/wildguard_audio_cnn_best.pt"
SAMPLES_DIR = ROOT / "outputs/demo_samples"

# Fixed 6 classes
CLASSES = [
    "Deforestation",
    "Forest_Environment",
    "Forest_Fire",
    "Gunshot_like",
    "Human_Sound",
    "Vehicle",
]

SR = 16000
DURATION = 4
N_SAMPLES = SR * DURATION
N_MELS = 64
N_FFT = 400
HOP = 160


# Model Architecture identical to training
class SmallCNN(torch.nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = SmallCNN(len(CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def preprocess_audio_bytes(audio_bytes, filename="audio.wav"):
    """Reads audio bytes, converts to 16kHz mono, pads/crops to 4s."""
    try:
        data, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception:
        # Fallback to librosa via temporary buffer
        audio_buf = io.BytesIO(audio_bytes)
        data, orig_sr = librosa.load(audio_buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        else:
            data = data.T

    audio = data.mean(axis=1)

    if orig_sr != SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)

    raw_duration = len(audio) / SR
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    is_silent = (peak < 1e-4) or (rms < 1e-5)

    if len(audio) > N_SAMPLES:
        audio_window = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio_window = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio_window = audio

    mel = librosa.feature.melspectrogram(
        y=audio_window,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=40,
        fmax=SR // 2,
        center=False,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80, 0) / 80.0
    tensor = torch.from_numpy(logmel.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return tensor, audio_window, logmel, raw_duration, peak, rms, is_silent


def render_waveform_and_spectrogram(audio_window, logmel):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.5), gridspec_kw={"height_ratios": [1, 1.2]})
    times = np.linspace(0, DURATION, len(audio_window))
    ax1.plot(times, audio_window, color="#1f77b4", linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, DURATION)
    ax1.grid(True, linestyle="--", alpha=0.5)

    im = ax2.imshow(
        logmel,
        origin="lower",
        aspect="auto",
        extent=[0, DURATION, 0, N_MELS],
        cmap="magma",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mel Band")
    fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.3, aspect=40, label="Norm Log-Mel")

    plt.tight_layout()
    return fig


# Streamlit Page Setup
st.set_page_config(
    page_title="WILDGUARD — Acoustic Intelligence",
    page_icon="🌲",
    layout="wide",
)

st.title("🌲 WILDGUARD — Forest Acoustic Intelligence Demo")
st.markdown(
    "Computer-based environmental acoustic monitoring system for forest protection. "
    "Classifies ambient and threat sounds across **6 target audio classes** using an edge-optimized Micro-CNN."
)

# Sidebar
st.sidebar.header("System Configuration")
st.sidebar.markdown(f"**Target Classes**: {len(CLASSES)}")
for c in CLASSES:
    st.sidebar.markdown(f"- `{c}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Audio Pipeline Spec**:")
st.sidebar.markdown(f"- Sample Rate: `{SR} Hz` (Mono)")
st.sidebar.markdown(f"- Window: `{DURATION} seconds` ({N_SAMPLES} samples)")
st.sidebar.markdown(f"- Log-Mel: `{N_MELS} bands`, FFT=`{N_FFT}`, Hop=`{HOP}`")
st.sidebar.markdown("- Micro-CNN params: `~26k params (105 KB)`")
st.sidebar.markdown("---")
st.sidebar.info("Model: `hard_negative_rerun` (Acc: 91.19%, Macro F1: 77.40%)")

# Load model
try:
    model, ckpt = load_model()
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model checkpoint: {e}")
    st.stop()

# Input Mode: Upload or Preset Samples
st.subheader("1. Audio Input")
input_mode = st.radio("Select input source:", ["Preset Demo Samples", "Upload Custom Audio File"], horizontal=True)

audio_bytes = None
audio_name = ""

if input_mode == "Preset Demo Samples":
    sample_files = sorted(list(SAMPLES_DIR.glob("*.wav"))) if SAMPLES_DIR.exists() else []
    sample_names = [f.name for f in sample_files]
    if sample_names:
        selected_sample = st.selectbox("Choose a sample audio clip:", sample_names)
        sample_path = SAMPLES_DIR / selected_sample
        audio_bytes = sample_path.read_bytes()
        audio_name = selected_sample
    else:
        st.warning("No preset samples found in outputs/demo_samples.")
else:
    uploaded = st.file_uploader("Upload audio file (WAV, MP3, FLAC, OGG):", type=["wav", "mp3", "flac", "ogg"])
    if uploaded is not None:
        audio_bytes = uploaded.read()
        audio_name = uploaded.name

if audio_bytes:
    st.audio(audio_bytes)

    # Process Audio
    with st.spinner("Processing audio features..."):
        try:
            tensor, audio_window, logmel, raw_dur, peak, rms, is_silent = preprocess_audio_bytes(audio_bytes, audio_name)
        except Exception as e:
            st.error(f"Error reading or decoding audio file: {e}")
            st.stop()

    st.markdown(f"**File**: `{audio_name}` | **Raw Duration**: `{raw_dur:.2f}s` | **Peak Amplitude**: `{peak:.4f}` | **RMS**: `{rms:.5f}`")

    # Handle Silence Alert
    if is_silent:
        st.warning("⚠️ **Warning: Audio is silent or near-silent!** Predictions on silent audio are uninformative.")

    # Model Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    top_prob = probs[top_idx]

    # Results Layout
    st.subheader("2. Classification Results")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Alert box color based on class
        if top_class in ["Deforestation", "Gunshot_like"]:
            alert_type = "error"
            badge = "🚨 THREAT DETECTED"
        elif top_class in ["Vehicle", "Human_Sound"]:
            alert_type = "warning"
            badge = "⚠️ SUSPICIOUS ACTIVITY"
        elif top_class == "Forest_Fire":
            alert_type = "warning"
            badge = "🔥 COMBUSTION SOUND"
        else:
            alert_type = "success"
            badge = "🌿 AMBIENT SOUND"

        st.metric(
            label=f"Predicted Class ({badge})",
            value=f"{top_class}",
            delta=f"Confidence: {top_prob*100:.1f}%",
        )

        # Context Guidance Notes
        if top_class in ["Human_Sound", "Vehicle"]:
            st.info(
                "ℹ️ **Operational Guidance**: Suspicious human vocal or vehicle engine sounds are **indicators** of "
                "potential activity, not definitive proof of unauthorized intrusion. Verification protocol required."
            )
        elif top_class == "Forest_Fire":
            st.info(
                "ℹ️ **Sensor Integration Note**: Audio acoustic fire detection acts as an auxiliary signal. "
                "Primary forest fire alerts remain sensor/rule-based (temperature, smoke, optical flame)."
            )

        st.markdown("#### Class Probabilities:")
        for idx, (cls, prob) in enumerate(zip(CLASSES, probs)):
            col_lbl, col_bar = st.columns([2, 5])
            with col_lbl:
                st.write(f"**{cls}**")
            with col_bar:
                st.progress(float(prob), text=f"{prob*100:.2f}%")

    with col2:
        st.markdown("#### Waveform & Log-Mel Spectrogram")
        fig = render_waveform_and_spectrogram(audio_window, logmel)
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("Select a preset demo sample above or upload an audio clip to begin acoustic inference.")
