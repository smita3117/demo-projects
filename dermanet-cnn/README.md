# 🩺 Dermatological Skin Condition Classification using CNNs

An end-to-end computer vision and deep learning masterclass focused on multi-class classification of clinical skin condition images using the DermNet dataset. This project progresses from first-principles Convolutional Neural Network (CNN) architecture design to transfer learning, addressing real-world medical imaging challenges such as dataset imbalance, ethical clinical considerations, and diagnostic evaluation metrics.

---

## 📌 Project Overview

* **Domain:** Computer Vision / Medical Image Analysis & Diagnostics
* **Dataset:** [DermNet Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
* **Task:** Multi-class classification across 23 distinct categories of skin diseases
* **Core Stack:** Python 3.12+, TensorFlow 2.x, Keras, NumPy, Pandas, Matplotlib, Seaborn, Kaggle API

---

## 📊 Dataset Breakdown

The dataset contains clinical photographs organized into pre-split train and test sets:

| Metric | Training Set | Test Set | Total |
| :--- | :--- | :--- | :--- |
| **Total Images** | 15,557 | 4,002 | 19,559 |
| **Classes** | 23 | 23 | 23 |

### ⚖️ Class Distribution & Imbalance
The dataset exhibits significant long-tailed class imbalance:

* **Top 5 Majority Classes:**
  1. *Psoriasis pictures Lichen Planus and related diseases* (1,405 images)
  2. *Seborrheic Keratoses and other Benign Tumors* (1,371 images)
  3. *Tinea Ringworm Candidiasis and other Fungal Infections* (1,300 images)
  4. *Eczema Photos* (1,235 images)
  5. *Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions* (1,149 images)
* **Least Represented Classes:**
  * *Urticaria Hives* (212 images)
  * *Hair Loss Photos Alopecia and other Hair Diseases* (239 images)
  * *Poison Ivy Photos and other Contact Dermatitis* (260 images)
  * *Cellulitis Impetigo and other Bacterial Infections* (288 images)

> **Clinical Implication:** Left unaddressed, standard cross-entropy loss causes the network to bias toward majority classes, generating deceptively high accuracy while failing on rare pathologies. The workflow mitigates this via targeted data augmentation, class-weight penalization, and stratified evaluation.

---

## 🔬 Pipeline Architecture & Methodology

The notebook follows a 7-stage applied machine learning lifecycle:

1. **Data Acquisition & Ingestion:** Secure authentication via Kaggle API / Colab Secrets, automated archive streaming (`dermnet.zip`), and extraction.
2. **Exploratory Data Analysis (EDA):** Class distribution inspection, Seaborn bar plot visualizations, and image artifact evaluation.
3. **CNN Fundamentals:** Handcrafted implementation of core vision building blocks (convolutions, padding, stride, pooling) from first principles.
4. **Baseline Modeling:** Building and training an initial CNN to establish an honest reference benchmark.
5. **Model Diagnostics:** Training curve interpretation, residual analysis, and overfitting inspection.
6. **Transfer Learning & Productionization:** Fine-tuning pre-trained backbones, class weight adjustments, and callback infrastructure.
7. **Clinical Deployment & Evaluation:** Multi-class confusion matrices, precision-recall analysis, and deployment considerations.

---

## 📁 Repository Structure

```text
dermnet-cnn/
├── README.md
├── notebooks/
│   └── CNN_Image_Recognition_Demo.ipynb
└── data/
    └── dermnet_data/
        ├── train/
        └── test/
⚙️ Environment Setup & Installation
1. Prerequisites
Python 3.10+ (Recommended: Python 3.12)

GPU-enabled runtime (NVIDIA CUDA GPU or Google Colab T4/V100/A100)

2. Dependency Installation
Bash
pip install tensorflow numpy pandas matplotlib seaborn kaggle tqdm
3. Kaggle API Configuration
Go to your Kaggle Account Settings and click Create New API Token to download kaggle.json.

Move it to the default directory:

Bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
(Alternatively, in Google Colab, add your keys to the Secrets tab as KAGGLE_USERNAME and KAGGLE_KEY.)
