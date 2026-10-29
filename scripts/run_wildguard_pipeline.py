"""
WILDGUARD Master Pipeline Orchestrator
One-command reproducible entry point for autonomous dataset engineering,
duplicate auditing, group-safe splitting, validation, and TinyML feature preparation.
Usage:
    python scripts/run_wildguard_pipeline.py
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPTS = [
    ("Audio File & Format Validation", "validate_audio_integrity.py"),
    ("Canonical Training Manifest Generation", "create_canonical_training_manifest.py"),
    ("Class & Group Distribution Report", "generate_class_distribution_report.py"),
    ("Automated Manifest & Zero-Leakage Tests", "test_manifest_validation.py"),
    ("TinyML Preprocessing & Spec Tests", "test_tinyml_preprocessing.py"),
]

def run_step(name, script_name, scripts_dir):
    script_path = scripts_dir / script_name
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print(f"Script: {script_path}")
    print("=" * 80)
    
    cmd = [sys.executable, "-u", str(script_path)]
    result = subprocess.run(cmd, cwd=str(scripts_dir))
    
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline step failed at: {name} (Exit code: {result.returncode})")
        sys.exit(result.returncode)
        
    print(f"[SUCCESS] Completed: {name}")

def main():
    start_time = datetime.now()
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    
    print("=" * 80)
    print("STARTING WILDGUARD REPRODUCIBLE DATASET & TINYML PIPELINE")
    print(f"Start time: {start_time.isoformat(timespec='seconds')}")
    print(f"Base Directory: {base_dir}")
    print("=" * 80)
    
    for name, script_file in SCRIPTS:
        run_step(name, script_file, scripts_dir)
        
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 80)
    print("WILDGUARD PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total pipeline execution time: {elapsed}")
    print("=" * 80)

if __name__ == "__main__":
    main()
