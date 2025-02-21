import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    images_dir = f"{input_dir}/images"
    labels_dir = f"{input_dir}/labels"
    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/test/images", exist_ok=True)
    os.makedirs(f"{output_dir}/test/labels", exist_ok=True)

    # List all images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(image_files)

    # Calculate split sizes
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split into train, val, test
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Copy files
    for f in train_files:
        shutil.copy(f"{images_dir}/{f}", f"{output_dir}/train/images/{f}")
        label = f.replace(".jpg", ".txt")
        if os.path.exists(f"{labels_dir}/{label}"):
            shutil.copy(f"{labels_dir}/{label}", f"{output_dir}/train/labels/{label}")
    
    for f in val_files:
        shutil.copy(f"{images_dir}/{f}", f"{output_dir}/val/images/{f}")
        label = f.replace(".jpg", ".txt")
        if os.path.exists(f"{labels_dir}/{label}"):
            shutil.copy(f"{labels_dir}/{label}", f"{output_dir}/val/labels/{label}")
    
    for f in test_files:
        shutil.copy(f"{images_dir}/{f}", f"{output_dir}/test/images/{f}")
        label = f.replace(".jpg", ".txt")
        if os.path.exists(f"{labels_dir}/{label}"):
            shutil.copy(f"{labels_dir}/{label}", f"{output_dir}/test/labels/{label}")

    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    split_dataset("data", "data/split")