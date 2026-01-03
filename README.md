# 🧬 Biofusion: Hybrid Biomedical Prediction

## 📌 Project Overview
**Biofusion Hybrid Biomedical Prediction** is a biomedical machine learning project focused on predicting diabetes status (diabetic vs. non-diabetic) from clinical, demographic, and laboratory features.

The primary objective of this phase is to develop a robust and interpretable supervised classification model using traditional machine learning techniques, while establishing a scalable foundation for future extensions involving regression tasks and deep learning–based hybrid models.
- Traditional machine learning
- Deep learning (Artificial Neural Networks)
- Classification, regression, and exploratory paradigms

Rather than a single isolated task, this project is designed as an **extensible framework**, where each phase builds upon the previous one while remaining independently reproducible.

---

## 🧠 Dataset
The project uses a clinical biomedical dataset containing demographic, laboratory, and physiological features.

At the beginning of the project, the dataset was split into:
- **Training set** (used for modeling and validation)
- **Test set** (held out and used only for final evaluation)

This ensures unbiased performance assessment across all phases.
> [Health Test by Blood Dataset](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)
---

## 📓 Notebooks (Current Phase)

### 1️⃣ Initial Inspection
**Notebook:** `Initial_Inspection.ipynb`

- First look at the dataset
- Shape, feature types, and basic statistics
- Data sanity checks
- Target variable inspection

---

### 2️⃣ Exploratory Data Analysis (EDA)
**Notebook:** `EDA.ipynb`

- Feature distribution visualization
- Skewness analysis and handling
- Outlier detection with medical-context awareness
- Feature filtering with minimal data loss (~1%)
- Log1p transformations for highly skewed variables
- Data quality validation before modeling

---

### 3️⃣ ML Classification
**Notebook:** `ML_Classification.ipynb`

- Supervised binary classification (Diabetic vs. Non-Diabetic)
- End-to-end preprocessing using pipelines
- Benchmarking of classic ML models
- Model selection based on **F1-weighted score**
- Hyperparameter tuning (GridSearch & RandomizedSearch)
- Final model selection, training, saving, and evaluation

**Final Model:** Support Vector Classifier (SVC)  
**Test F1-weighted Score:** ≈ **82.23%**

---

## 🗂 Project Structure

```
Biofusion-Hybrid-Biomedical-Prediction/
│
├── Data/
│ ├── dataset.csv
│ ├── train.csv
│ └── test.csv
│
├── Notebooks/
│ ├── Initial-Inspection.ipynb
│ ├── EDA.ipynb
│ └── ML-Classification.ipynb
│
├── Models/
│ └── ML-Classification/
│       └── ML-Cls-svc-model.pkl
│
├── Source/
│  ├── ML-Classification/
│     ├── preprocessor.py
│     └── trainer.py
│
├── requirements.txt
└── README.md

```


---

## 🧪 Methodological Highlights (Current Phase)
- Data leakage prevention via pipelines
- Medical-domain-aware preprocessing
- Robust evaluation using an untouched test set
- Emphasis on reproducibility and clarity over brute-force optimization

---

## 📊 Final Classification Results
- **F1-weighted:** ~82.23%
- **False Negatives:** ~120 samples
- **False Positives:** ~60 samples

These metrics provide a solid baseline for medical risk-sensitive refinement in later phases.

---

## 🚀 Planned Next Phases

### 🔹 Phase 2: ML Regression
- Predict continuous biomedical targets
- Traditional ML models
- ANN-based regression comparison

### 🔹 Phase 3: Hybrid Biofusion Deep Learning (ANN) Model
- Artificial Neural Networks for:
- Combined classification + regression framework
- Shared feature representations
- Unified hybrid modeling approach
- Toward an integrated biomedical prediction system

### 🔹 (Considering) Phase 4: Unsupervised Learning
- Clustering and latent structure discovery
- Exploratory rather than primary objective

---

## 🎯 Project Philosophy
This project prioritizes:
- **Depth over breadth**
- **High-quality, explainable models**
- **Research-readiness**
- **Incremental, well-documented progress**

Rather than many shallow projects, Biofusion is designed as a **single evolving biomedical research pipeline**.

---

## 🛠 Tools & Libraries
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

---

## 📌 Notes
Each phase is self-contained but intentionally designed to integrate with future phases, allowing this project to grow into a full-scale biomedical prediction framework.

