# Crime Detection Web App
A web application for detecting crime-related objects using YOLOv8 and MobileNet.

## Setup
1. Clone the repo: `git clone https://github.com/Pranaykumar30/crime-detection-webapp.git`
2. Activate virtual env: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python app/main.py`

## Progress
- **Day 1**: Project setup completed in GitHub Codespaces with Flask starter.
- **Day 2**: Dataset collection and preprocessing completed.
  - Manually uploaded 180 images (30 per class: handguns, knives, sharp-edged-weapons, masked-intruders, violence, normal-behavior).
  - Organized into YOLO format, auto-labeled with YOLOv8, stored raw data in data/images/ and data/labels/.
  - Split into train/val/test (70/15/15) in data/split/.
  - Applied advanced preprocessing: resize, normalization, geometric/color augmentations, noise/blur, edge enhancement, contrast stretching.