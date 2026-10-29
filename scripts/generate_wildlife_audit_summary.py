"""
WILDGUARD Wildlife Audio Audit Summary Generator
Documents:
1. Existing species and recording counts (Lion: 75, WolfHowl: 75)
2. Parent recording overlap / contiguous chunk audit (37 Lion parents, 22 WolfHowl parents)
3. New recordings located (Leopard: 7, Grizzly Bear: 1)
4. Label confidence, contamination risks (e.g. dog howl in WolfHowl)
5. Recommended additional datasets and species coverage
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

def main():
    base_dir = Path(r"C:\Users\Nischal N Bangera\OneDrive\Desktop\WILDGUARD")
    out_dir = base_dir / "data" / "processed"
    
    summary_text = f"""WILDGUARD WILDLIFE AUDIO AUDIT REPORT (PHASE 2 - WILD ANIMAL SPECIFIC)
Generated: {datetime.now().isoformat(timespec='seconds')}
Location: {base_dir}

================================================================================
EXECUTIVE SUMMARY
================================================================================
This audit specifically inspects the WILDGUARD codebase and raw datasets for genuine
wild-animal audio recordings. As instructed:
- Baseline model training is strictly PAUSED.
- The 150 baseline FSC22 clips (Lion, WolfHowl) are preserved untouched.
- Domestic animals (dogs, cats, pigs, cows, sheep, hens) and ordinary environmental
  creatures (birds, frogs, insects, squirrels) are strictly excluded from Wild_Animal.
- Physical audio, parent recording relationships, and label confidences have been verified.

================================================================================
1. EXISTING WILD-ANIMAL SPECIES & USABLE RECORDING COUNTS
================================================================================
Currently, only TWO wild-animal species exist in the active WILDGUARD dataset (both from FSC22):

1. Lion (Panthera leo)
   - Total clips: 75
   - Clip duration: 5.0 seconds each
   - Audio format: 16-bit WAV, variable sample rates (resampled to 16 kHz)
   - Parent Freesound source recordings: 37 unique parents
   - Multi-clip parent recordings: 20 parents were sliced into 58 clips (e.g., sound ID 24797 accounts for 4 slices)
   - True independent recording events: 37

2. Wolf Howl (Canis lupus)
   - Total clips: 75
   - Clip duration: 5.0 seconds each
   - Audio format: 16-bit WAV, variable sample rates (resampled to 16 kHz)
   - Parent Freesound source recordings: 22 unique parents
   - Multi-clip parent recordings: 11 parents account for 64 clips (e.g., sound ID 541282 accounts for 6 slices)
   - True independent recording events: 22

Total Current Baseline: 150 clips (59 unique parent acoustic events)

NON-WILD SPECIES VERIFIED EXCLUDED:
- FSC22: Squirrel (75 clips), Frog (75), Insect (75), BirdChirping (75) -> Assigned to Forest_Environment.
- ESC-50: Dog (40 clips), Cat (40), Pig (40), Cow (40), Sheep (40), Hen (40), Rooster (40) -> Strictly Domestic, excluded from Wild_Animal.
- ESC-50: Chirping birds (40), Crow (40), Frog (40), Insects (40), Crickets (40) -> Assigned to Forest_Environment.
- RFCx FrugalAI: 50,392 clips -> Canopy rainforest ambience and chainsaws only; no species-level tags.
- GunshotDataset: 2,148 clips -> Firearm discharges only.
- ForestWildFire: 289 clips -> Combustion noise only.
- ForestSoundDataset: 2,761 clips -> Forest ambience, logging, fire; no mammal species tags.

================================================================================
2. GENUINELY NEW RECORDINGS DISCOVERED IN WORKSPACE / DOWNLOADS
================================================================================
A scan of local directories located 8 verified, unintegrated wildlife recordings:

1. Leopard (Panthera pardus) - 7 recordings (Total duration: 179.3 seconds)
   - Source: Xeno-canto (verified academic bioacoustic database)
   - XC1077460 (20.7s, 48 kHz stereo, Panthera pardus pardus)
   - XC1077461 (18.0s, 48 kHz stereo, Panthera pardus pardus)
   - XC1098554 (22.5s, 48 kHz stereo, Panthera pardus pardus)
   - XC1098555 (26.9s, 48 kHz stereo, Panthera pardus pardus)
   - XC1098556 (23.9s, 48 kHz stereo, Panthera pardus fusca)
   - XC1149603 (55.5s, 48 kHz mono, Panthera pardus)
   - XC959813  (11.9s, 44.1 kHz stereo, Panthera pardus pardus)
   - Usable 4s chunks: ~35-40 standardized segments

2. Grizzly Bear (Ursus arctos horribilis) - 1 recording (Total duration: 92.0 seconds)
   - Source: Freesound (ID 519518, User nivatius)
   - 44.1 kHz, 1 channel mono, 91.98 seconds
   - Usable 4s chunks: ~18-20 standardized segments

================================================================================
3. EXACT DUPLICATES & POSSIBLE OVERLAPPING RECORDINGS
================================================================================
1. Exact File Duplicates (SHA-256):
   - Among the 150 baseline FSC22 clips, 150 unique SHA-256 hashes exist (0 exact file duplicates).
   - Zero hash overlap exists between the FSC22 clips and the 8 new downloaded recordings.

2. Contiguous / Overlapping Slices (CRITICAL DATA LEAKAGE FINDING):
   - In FSC22 Lion: 37 unique parents produced 75 clips. 58 clips share parent recordings.
   - In FSC22 WolfHowl: 22 unique parents produced 75 clips. 64 clips share parent recordings.
   - Audit Finding: In earlier split versions, 130 FSC22 parent recordings had slices split across
     train and validation because grouping was done by slice filename rather than parent Freesound ID.
   - Resolution: When finalizing the Wild_Animal dataset, clips must strictly be grouped by parent ID
     (e.g., Freesound ID '24797' or '128469') to prevent contiguous chunk leakage.

================================================================================
4. LABEL CONFIDENCE & DATASET LICENSING
================================================================================
1. FSC22 Lion:
   - License: CC BY 4.0 (Sensors 2023, Bandara et al.) / Freesound CC.
   - Label Confidence: High. Authentic lion roars and growls, though partially recorded in captive
     sanctuary settings (e.g. Lincoln Park Zoo).

2. FSC22 WolfHowl:
   - License: CC BY 4.0 / Freesound CC.
   - Label Confidence: Medium / Contaminated.
     * WARNING: Clip 26_12606.wav is derived from '66538__robinhood76__00839-dog-howl-1.wav',
       which is a domestic dog howl. It MUST be quarantined from Wild_Animal to respect the
       rule against domestic animal contamination.
     * Clip 26_12607.wav is derived from '640986__beautifuldaymonster1968__the-classic-wolf-tv-howl.wav'
       (foley/sound effect).

3. Xeno-canto Leopards:
   - License: Creative Commons (CC BY-NC-SA 4.0 / CC BY 4.0).
   - Label Confidence: Very High. Verified field bioacoustic recordings with collector metadata,
     geographic coordinates, and species identification.

4. Freesound Grizzly Bear:
   - License: CC0 / CC BY.
   - Label Confidence: High. Verified bear vocalization.

================================================================================
5. SUITABLE ADDITIONAL DATASETS & RECOMMENDATIONS FOR MISSING SPECIES
================================================================================
To build a robust Wild_Animal class capable of reliably triggering alerts on an ESP32-S3
without false alarms, the following key species should be integrated:

Target Wildlife Species to Add:
1. Tiger (Panthera tigris)
   - Vocalizations: Roars, chuffs, growls
   - Recommended Source: Xeno-canto (Panthera tigris) & India Ecoacoustics Network (IEN) Dataset (Zenodo 2026)

2. Elephant (Elephas maximus / Loxodonta africana)
   - Vocalizations: Trumpets, low-frequency rumbles, roars
   - Recommended Source: Cornell Elephant Listening Project (ELP) & Audio-Classification-for-Elephant-Sounds (Kaggle/GitHub)

3. Bear (Ursus arctos / Ursus americanus)
   - Vocalizations: Huffs, growls, roars
   - Recommended Source: Yellowstone National Park Sound Library (US NPS - Public Domain) & Freesound

4. Wild Boar (Sus scrofa)
   - Vocalizations: Grunts, squeals, rooting (distinct from domestic pigs)
   - Recommended Source: Xeno-canto & Tierstimmenarchiv

5. Deer / Elk (Cervus canadensis / Axis axis)
   - Vocalizations: Rutting calls, bugles, alarm barks
   - Recommended Source: US National Park Service (Public Domain) & Xeno-canto

6. Owls (Bubo virginianus / Tyto alba / Strix varia)
   - Vocalizations: Territorial hoots, screeches (distinct from songbirds)
   - Recommended Source: Xeno-canto

RECOMMENDED IMMEDIATE ACTION:
1. Quarantine clip 26_12606.wav (dog howl) from the WolfHowl baseline.
2. Group FSC22 clips strictly by parent Freesound ID to prevent intra-recording leakage.
3. Ingest the 7 verified Leopard recordings and 1 Grizzly Bear recording from Downloads.
4. Download targeted public domain / CC-BY samples for Elephant, Tiger, Deer, and Owl from Xeno-canto/NPS.
"""
    out_file = out_dir / "wildlife_audio_audit_summary_v1.txt"
    out_file.write_text(summary_text, encoding="utf-8")
    print(f"Audit summary successfully written to: {out_file}")

if __name__ == "__main__":
    main()
