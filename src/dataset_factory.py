import os
import numpy as np
from create_dataset import dataset_creation  # <-- adjust name if needed

# -------------------------------
# CONFIGURATION
# -------------------------------
SAMPLES_PER_DATASET = 2000     # each file contains 2000 samples
NUM_DATASETS = 40              # produces 20 files = 40,000 samples total
OUTPUT_FOLDER = "datasets_big2"
# -------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Starting overnight dataset generation:")
print(f" → {NUM_DATASETS} datasets")
print(f" → {SAMPLES_PER_DATASET} samples each")
print(f" → Total samples: {SAMPLES_PER_DATASET * NUM_DATASETS}")
print(f" → Output folder: {OUTPUT_FOLDER}\n")

for i in range(NUM_DATASETS):
    filename = os.path.join(OUTPUT_FOLDER, f"dataset_{i+1}.npz")

    print(f"[{i+1}/{NUM_DATASETS}] Generating {filename} ...")

    try:
        dataset_creation(
            num_tests=SAMPLES_PER_DATASET,
            save_path=filename
        )
        print(f"✓ Saved {filename}\n")

    except Exception as e:
        print(f"🔥 ERROR generating dataset {i+1}: {e}")
        print("Skipping to next...\n")
        continue

print("🎉 All datasets completed!")