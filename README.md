# Waste Image Classification: ResNet50 vs EfficientNetB3 vs YOLOv11

A UTS project comparing three transfer-learning approaches for sorting waste images into six classes:
biodegradable, cardboard, glass, metal, paper and plastic.

## Notebooks

| Notebook | Approach |
|---|---|
| `01_yolov11_detection.ipynb` | YOLOv11 nano object detector (Ultralytics), 10 epochs |
| `02_resnet50_classifier.ipynb` | ResNet50 classifier (TensorFlow/Keras), ImageNet weights, two-phase training |
| `03_efficientnetb3_classifier.ipynb` | EfficientNetB3 classifier (TensorFlow/Keras), ImageNet weights, two-phase training |

## Data

- Roboflow waste dataset, 17,365 images across six classes.
- The original splits were missing whole classes, so the images were reorganised into class folders and
  re-split 70/20/10: 12,152 train, 3,470 validation and 1,743 test images.
- The dataset is not included. Set your own Roboflow key where the notebooks say `YOUR_ROBOFLOW_API_KEY`.

## Method

- **Classifiers:** frozen ImageNet base with a small classification head (10 epochs), then the top layers
  unfrozen and fine-tuned at a learning rate of 1e-5 (15 epochs), with image augmentation.
- **Detector:** pre-trained `yolo11n.pt` fine-tuned on the YOLO-format export of the same dataset.

## Results

| Model | Result |
|---|---|
| YOLOv11n | mAP50 0.47, mAP50-95 0.32, precision 0.53, recall 0.42 |
| ResNet50 | best validation accuracy 0.56, top-3 accuracy 0.86 |
| EfficientNetB3 | test accuracy 0.37, top-3 accuracy 0.71 |

## Known issues and next steps

- Both classifiers feed images scaled to 0-1 (`rescale=1./255`). Keras EfficientNet models expect raw 0-255
  pixels because they rescale internally, and ResNet50 expects `preprocess_input`. This mismatch is the most
  likely reason the classifiers underperform, and fixing it is the first next step.
- Train for longer and report per-class results on the held-out test set for all three models.

## Tools

Python, TensorFlow/Keras, Ultralytics YOLO, Roboflow, scikit-learn, OpenCV, Google Colab.
