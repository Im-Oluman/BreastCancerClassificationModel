# 🧠 Breast Cancer Classification ML Project

This project builds and compares machine learning models to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using patient diagnostic data.

---

##  Dataset

- Source: Breast Cancer Wisconsin Diagnostic Dataset
- Target: `diagnosis`
  - M → Malignant (1)
  - B → Benign (0)

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Feature Engineering
4. Label Encoding
5. Train-Test Split
6. Model Training
   - Random Forest
   - Logistic Regression
7. Model Evaluation
8. Cross Validation
9. Model Saving

---

##  Models Used

### 1. Logistic Regression
- Interpretable baseline model
- Requires feature scaling

### 2. Random Forest Classifier
- High performance ensemble model
- Handles non-linear relationships

---

##  Evaluation Metrics

- Accuracy Score
- Classification Report
- Cross Validation Score (5-fold)

---

##  Results Summary

| Model | Accuracy | CV Score |
|------|--------|---------|
| Logistic Regression | ~95–98% | ~95–97% |
| Random Forest | ~96–99% | ~96–99% |

---

##  Project Structure
BREAST CANCER CLASSIFICATION MODEL/
│
├── data/
│ └── data.csv
│
├── models/
│ ├── random_forest_model.pkl
│ 
│
├── notebooks/
│ └── Breast Cancer.ipynb
│
├── main.py
├── README.md
├── requirements.txt 