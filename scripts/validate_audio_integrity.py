"""
WILDGUARD Audio Integrity Validation
Phase 2A - Validates physical audio file readability, format, duration, sample rate, and channels.
Does NOT modify or move any audio files.
Outputs:
- data/processed/audio_validation_report_v1.csv (summary and anomalies)
- data/processed/missing_corrupted_audio_summary_v1.txt
"""
import os
import sys
import time
import wave
from pathlib import Path
from datetime import datetime
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def main():
    base_dir = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
    manifest_path = base_dir / "data" / "processed" / "master_manifest_v10_split_plan.csv"
    out_dir = base_dir / "data" / "processed"
    
    if not manifest_path.exists():
        print(f"Error: manifest {manifest_path} not found.")
        sys.exit(1)
        
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, low_memory=False)
    total_files = len(df)
    print(f"Total audio files to validate: {total_files}")
    
    anomalies = []
    dataset_stats = {}
    
    start_time = time.time()
    valid_count = 0
    missing_count = 0
    corrupt_count = 0
    empty_count = 0
    
    for idx, row in df.iterrows():
        p = str(row["audio_path"])
        ds = str(row["dataset"])
        
        if ds not in dataset_stats:
            dataset_stats[ds] = {
                "total": 0, "valid": 0, "missing": 0, "corrupt": 0, "empty": 0,
                "sample_rates": set(), "channels": set(), "durations": []
            }
        dataset_stats[ds]["total"] += 1
        
        if not os.path.exists(p):
            missing_count += 1
            dataset_stats[ds]["missing"] += 1
            anomalies.append({
                "audio_path": p,
                "dataset": ds,
                "wildguard_class": row.get("wildguard_class", ""),
                "status": "MISSING",
                "detail": "File not found on disk"
            })
            continue
            
        file_size = os.path.getsize(p)
        if file_size == 0:
            empty_count += 1
            dataset_stats[ds]["empty"] += 1
            anomalies.append({
                "audio_path": p,
                "dataset": ds,
                "wildguard_class": row.get("wildguard_class", ""),
                "status": "EMPTY",
                "detail": "File size is 0 bytes"
            })
            continue
            
        # Attempt wave header read
        read_ok = False
        ch = None
        sr = None
        dur = None
        
        try:
            with wave.open(p, 'rb') as wf:
                ch = wf.getnchannels()
                sr = wf.getframerate()
                frames = wf.getnframes()
                dur = frames / sr if sr > 0 else 0.0
                read_ok = True
        except Exception:
            # Fallback to soundfile for non-standard WAV or other formats
            try:
                info = sf.info(p)
                ch = info.channels
                sr = info.samplerate
                dur = info.duration
                read_ok = True
            except Exception as e:
                read_ok = False
                corrupt_count += 1
                dataset_stats[ds]["corrupt"] += 1
                anomalies.append({
                    "audio_path": p,
                    "dataset": ds,
                    "wildguard_class": row.get("wildguard_class", ""),
                    "status": "CORRUPTED",
                    "detail": str(e)
                })
                
        if read_ok:
            valid_count += 1
            dataset_stats[ds]["valid"] += 1
            dataset_stats[ds]["sample_rates"].add(sr)
            dataset_stats[ds]["channels"].add(ch)
            if len(dataset_stats[ds]["durations"]) < 500: # sample up to 500 for stats
                dataset_stats[ds]["durations"].append(dur)
                
            if dur is not None and dur < 0.1:
                anomalies.append({
                    "audio_path": p,
                    "dataset": ds,
                    "wildguard_class": row.get("wildguard_class", ""),
                    "status": "EXTREMELY_SHORT",
                    "detail": f"Duration {dur:.3f}s is under 0.1s"
                })
                
        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Validated {idx + 1}/{total_files} files ({elapsed:.1f}s, {(idx+1)/elapsed:.1f} files/s)...")
            
    total_time = time.time() - start_time
    print(f"\nCompleted audio validation in {total_time:.1f}s.")
    print(f"Valid: {valid_count} | Missing: {missing_count} | Empty: {empty_count} | Corrupted: {corrupt_count}")
    
    # Save anomalies report
    anomalies_df = pd.DataFrame(anomalies)
    anom_file = out_dir / "audio_validation_report_v1.csv"
    anomalies_df.to_csv(anom_file, index=False, encoding="utf-8-sig")
    print(f"Saved anomaly report to {anom_file} ({len(anomalies_df)} rows)")
    
    # Generate summary text
    summary_lines = [
        "WILDGUARD AUDIO INTEGRITY VALIDATION REPORT",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input manifest: {manifest_path}",
        "",
        f"Total audio files audited: {total_files}",
        f"Successfully readable:     {valid_count} ({valid_count/total_files*100:.2f}%)",
        f"Missing files:             {missing_count}",
        f"Empty (0-byte) files:      {empty_count}",
        f"Corrupted/unreadable files:{corrupt_count}",
        f"Total anomalies:           {len(anomalies_df)}",
        "",
        "DATASET AUDIT BREAKDOWN:",
        "-" * 70,
        f"{'Dataset':<22} {'Total':<8} {'Valid':<8} {'Channels':<12} {'Sample Rates':<18}"
    ]
    for ds, st in dataset_stats.items():
        srs = ", ".join(str(s) for s in sorted(st["sample_rates"]))
        chs = ", ".join(str(c) for c in sorted(st["channels"]))
        summary_lines.append(f"{ds:<22} {st['total']:<8} {st['valid']:<8} {chs:<12} {srs:<18}")
    summary_lines.extend([
        "-" * 70,
        "",
        "VERIFICATION STATUS:",
        "PASS: All existing manifest audio files are verified physically readable." if (missing_count == 0 and corrupt_count == 0 and empty_count == 0) else "WARNING: Anomalies found, see audio_validation_report_v1.csv",
        "",
        f"Anomaly report path: {anom_file}"
    ])
    
    summary_path = out_dir / "missing_corrupted_audio_summary_v1.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved summary to {summary_path}")
    print("\n" + "\n".join(summary_lines))

if __name__ == "__main__":
    main()
