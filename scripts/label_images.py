from ultralytics import YOLO
import os

def label_images(image_dir, label_dir):
    model = YOLO("yolov8n.pt")  # Pretrained nano model
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for img in image_files:
        results = model.predict(f"{image_dir}/{img}", save_txt=True)
        label_file = img.replace(".jpg", ".txt")
        if os.path.exists(f"runs/detect/predict/labels/{label_file}"):
            os.rename(f"runs/detect/predict/labels/{label_file}", f"{label_dir}/{label_file}")

if __name__ == "__main__":
    os.makedirs("data/labels", exist_ok=True)
    label_images("data/images", "data/labels")