import os
import shutil
import random
from pathlib import Path

# Original dataset directory
SOURCE_DIR = Path("data")

# New small dataset destination
DEST_DIR = Path("data_sample")

# Number of images to copy per class
SPLIT_COUNTS = {
    "train": 1000,
    "valid": 200,
    "test": 200
}

CLASSES = ["fake", "real"]

def create_sample_split():
    for split in SPLIT_COUNTS:
        for cls in CLASSES:
            src_folder = SOURCE_DIR / split / cls
            dest_folder = DEST_DIR / split / cls

            # Make destination directory
            dest_folder.mkdir(parents=True, exist_ok=True)

            # Get all image filenames and shuffle them
            all_files = list(src_folder.glob("*.jpg"))  # or .png based on dataset
            random.shuffle(all_files)

            # Copy only a small number
            for file in all_files[:SPLIT_COUNTS[split]]:
                shutil.copy(file, dest_folder / file.name)

            print(f"Copied {SPLIT_COUNTS[split]} images to {dest_folder}")

if __name__ == "__main__":
    create_sample_split()
