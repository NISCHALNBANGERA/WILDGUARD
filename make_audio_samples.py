from pathlib import Path
import soundfile as sf
import numpy as np

src = Path(r"C:\Users\Nischal N Bangera\Downloads\audio\audiomoth\Arden_20250606_130550.WAV")
out = Path(r"C:\Users\Nischal N Bangera\Downloads\audio\audiomoth_samples")
out.mkdir(exist_ok=True)

info = sf.info(str(src))
starts = [0, max(0, info.duration / 2 - 10), max(0, info.duration - 20)]

for name, start in zip(["beginning", "middle", "end"], starts):
    audio, sr = sf.read(
        str(src),
        start=int(start * info.samplerate),
        frames=min(20 * info.samplerate, info.frames - int(start * info.samplerate)),
        dtype="float32"
    )
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.8
    sf.write(str(out / f"{name}.wav"), audio, sr, subtype="PCM_16")
    print(f"{name}: {len(audio)/sr:.1f} seconds; original peak={peak:.6f}")

print(f"Samples saved to: {out}")
