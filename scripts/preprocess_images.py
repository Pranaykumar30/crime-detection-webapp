import cv2
import numpy as np
import albumentations as A
import os
from skimage import exposure, filters

def preprocess_images(image_dir, output_dir):
    """Apply advanced preprocessing to images."""
    os.makedirs(output_dir, exist_ok=True)

    # Updated augmentation pipeline for modern albumentations
    augment = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # Updated RandomSizedCrop syntax
        A.RandomCrop(height=640, width=640, p=0.4),  # Simpler crop to fixed size
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ToGray(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2)
    ])

    for img_file in os.listdir(image_dir):
        img_path = f"{image_dir}/{img_file}"
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Step 1: Resize to consistent dimensions
        img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

        # Step 2: Normalize pixel values (0-1 range)
        img_normalized = img_resized / 255.0

        # Step 3: Apply augmentations
        augmented = augment(image=(img_normalized * 255).astype(np.uint8))
        img_aug = augmented["image"]

        # Step 4: Enhance edges with Sobel filter
        img_gray = cv2.cvtColor(img_aug, cv2.COLOR_BGR2GRAY)
        edges = filters.sobel(img_gray)
        img_edges = cv2.convertScaleAbs(edges * 255)
        img_aug = cv2.addWeighted(img_aug, 0.8, cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR), 0.2, 0)

        # Step 5: Contrast stretching
        img_aug = exposure.rescale_intensity(img_aug, in_range="image", out_range=(0, 255)).astype(np.uint8)

        # Save processed image
        cv2.imwrite(f"{output_dir}/{img_file}", img_aug)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        input_dir = f"data/split/{split}/images"
        preprocess_images(input_dir, input_dir)
        print(f"Preprocessed {split} images")