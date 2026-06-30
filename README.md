# Face Recognition CNN — LFW Dataset

**Ayush Ranjan | 23MIP10135 | VIT Bhopal**

## Results
| Metric | Score |
|---|---|
| Validation Accuracy | **86.23%** |
| Top-2 Accuracy | **94.61%** |
| Parameters | 2,758,191 |

## Dataset
- **Source:** [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) — Kaggle
- **Filtered to:** Top 15 individuals with ≥ 20 images each
- **Total images:** 1,701 (1,367 train / 334 val)
- **Image size:** 128 × 128 px

### Selected Individuals
George W Bush · Colin Powell · Tony Blair · Donald Rumsfeld · Gerhard Schroeder · Ariel Sharon · Hugo Chavez · Junichiro Koizumi · Jean Chretien · John Ashcroft · Serena Williams · Jacques Chirac · Vladimir Putin · Gloria Macapagal Arroyo · others

## Architecture — Deep CNN (5 Conv Blocks)
```
Input (128×128×3)
├── Block 1: Conv2D(32)×2  → BatchNorm → MaxPool → Dropout(0.20)
├── Block 2: Conv2D(64)×2  → BatchNorm → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128)×2 → BatchNorm → MaxPool → Dropout(0.30)
├── Block 4: Conv2D(256)×2 → BatchNorm → MaxPool → Dropout(0.30)
├── Block 5: Conv2D(512)   → BatchNorm            → Dropout(0.30)
├── GlobalAveragePooling2D
├── Dense(512) → BatchNorm → Dropout(0.50)
├── Dense(256) → Dropout(0.40)
└── Dense(15, softmax)
```
All Conv layers: ReLU · same-padding · L2(1e-4) regularization

## Training Config
| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, EarlyStopping) |
| Batch Size | 16 |
| Image Size | 128×128 |
| Augmentation | rotation ±25°, shift, zoom, h-flip, brightness |
| Framework | TensorFlow 2.20 / Keras |
| Hardware | Google Colab T4 GPU |

## How to Run
1. Open `1_Face_Recognition_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime → **T4 GPU**
3. Run all cells top to bottom
