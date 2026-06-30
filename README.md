# Face Recognition using CNN

> **Assignment Submission — VIT Bhopal**

| Field | Details |
|---|---|
| **Name** | Ayush Ranjan |
| **Roll No** | 23MIP10135 |
| **Branch** | MIP |
| **Institute** | VIT Bhopal |

---

## Project Overview

This project implements a **Face Recognition system** using a custom deep Convolutional Neural Network (CNN) trained on the **Labeled Faces in the Wild (LFW)** dataset from Kaggle.

The model identifies individuals from face images, trained on the top 15 most-represented people (≥ 20 images each) in the LFW dataset.

---

## Dataset

**Labeled Faces in the Wild (LFW)**
- Source: [Kaggle — jessicali9530/lfw-dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- 5,749 individuals, 13,233 total images
- We filter to individuals with ≥ 20 images (top 15 people)
- Images: 250×250 px (resized to 128×128 for training)

---

## CNN Architecture

```
Input (128×128×3)
    ↓
Block 1: Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Block 2: Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128) × 2 → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Block 4: Conv2D(256) × 2 → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Block 5: Conv2D(512) → BatchNorm → Dropout(0.3)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.4)
    ↓
Dense(num_classes, softmax)
```

**Regularisation:** L2 weight decay (1e-4) on all conv/dense layers, BatchNormalization after every conv layer, Dropout throughout.

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Batch Size | 16 |
| Max Epochs | 60 (EarlyStopping) |
| Image Size | 128×128 |
| Augmentation | Rotation, shift, flip, zoom, brightness, shear |

---

## How to Run

1. Open `1_Face_Recognition_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells top to bottom
4. The Kaggle API is pre-configured in the notebook

---

## Files

```
├── 1_Face_Recognition_CNN.ipynb    # Main Colab notebook
└── README.md
```

---

## Results

The model outputs:
- Validation accuracy & Top-2 accuracy
- Training/validation accuracy and loss curves
- Confusion matrix
- Per-class classification report
- Sample predictions with confidence scores
