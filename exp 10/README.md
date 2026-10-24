# 🌸 Iris Flower Classification Using K-Nearest Neighbors (KNN)

A complete, working machine learning mini-project and local web application built with **Python**, **Flask**, and **Scikit-Learn** for college ML laboratory demonstrations and project records.

---

## 1. Project Description & Objective

The objective of this project is to build an end-to-end Machine Learning web application that predicts the biological species of an Iris flower based on its petal and sepal physical measurements.

### Predicted Flower Species (Output Classes):
1. **Setosa** (*Iris setosa*)
2. **Versicolor** (*Iris versicolor*)
3. **Virginica** (*Iris virginica*)

### Morphological Input Features:
1. **Sepal Length (cm)**
2. **Sepal Width (cm)**
3. **Petal Length (cm)**
4. **Petal Width (cm)**

---

## 2. Dataset Information

The project uses the **Iris dataset built directly into Scikit-learn** (`sklearn.datasets.load_iris()`). 
> **Note:** The dataset is supplied by Scikit-Learn; no external APIs or synthetic datasets are used.

- **Total Samples:** 150 biological specimens
- **Samples per Species:** 50 Setosa, 50 Versicolor, 50 Virginica (perfectly balanced)
- **Features:** 4 continuous numerical dimensions
- **Local Copy:** Saved as `data/iris.csv`

---

## 3. Complete 10-Step Machine Learning Workflow

This project adheres to the complete machine learning lifecycle, suitable for laboratory demonstration:

1. **Problem Definition:** Supervised 3-class classification of Iris flowers from continuous morphological measurements.
2. **Data Collection:** Load Scikit-learn's built-in dataset and export a local CSV copy (`data/iris.csv`).
3. **Data Preprocessing:** Verify missing values (0 found), inspect duplicate rows, check data types (float64), and separate features $X$ from target $y$.
4. **Exploratory Data Analysis (EDA):** Generate genuine visualizations using Matplotlib & Seaborn saved to `static/plots/`.
5. **Feature Selection:** Utilize all 4 features (Sepal Length, Sepal Width, Petal Length, Petal Width) as each contributes distinct physical separation.
6. **Train-Test Split:** Stratified 80% training set (120 samples) and 20% testing set (30 samples) with fixed `random_state=42`.
7. **Feature Scaling:** Apply `StandardScaler` fitted **strictly on the training data** to prevent data leakage.
8. **KNN Model Training:** Assemble a Scikit-Learn `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])`. 5-fold cross-validation on training data selects the optimal $K=5$.
9. **Model Evaluation:** Evaluate the fitted pipeline on the unseen test set (30 samples), reporting Accuracy, Precision, Recall, F1-score (macro), and Confusion Matrix.
10. **Prediction on New Input:** An interactive web form validates inputs and outputs the predicted species and model-estimated class probabilities.

---

## 4. K-Nearest Neighbors (KNN) Algorithm & Scaling

- **Algorithm:** K-Nearest Neighbors is an instance-based, non-parametric algorithm that assigns a query point to the majority class among its $K$ closest training samples in Euclidean space:
  $$d(\mathbf{x}, \mathbf{p}) = \sqrt{\sum_{j=1}^{4} (x_j - p_j)^2}$$
- **Hyperparameter:** $K = 5$ neighbors selected via 5-fold cross-validation on training folds.
- **Feature Scaling:** Since KNN calculates distance, features with larger scales would disproportionately dominate. `StandardScaler` standardizes each feature:
  $$z = \frac{x - \mu}{\sigma}$$
  The scaler is bundled with the classifier in a Scikit-learn `Pipeline` to ensure uniform preprocessing during inference.

---

## 5. Model Evaluation Metrics (Test Set Results)

- **Test Accuracy:** 93.33% (28 / 30 correct)
- **Macro Precision:** 94.44%
- **Macro Recall:** 93.33%
- **Macro F1-Score:** 93.27%
- **Confusion Matrix:**
  - Setosa: 10 / 10 correct (100%)
  - Versicolor: 10 / 10 correct (100%)
  - Virginica: 8 / 10 correct (2 borderline samples near Versicolor boundary)

---

## 6. Project Directory Structure

```text
IrisFlowerClassification/
├── app.py                      # Flask web application & prediction handler
├── train_model.py              # Complete ML workflow, EDA generation & model trainer
├── requirements.txt            # Required Python dependencies
├── README.md                   # Project documentation & laboratory guide
├── data/
│   └── iris.csv                # Local CSV copy of Scikit-Learn Iris dataset
├── models/
│   ├── iris_knn_model.pkl      # Persisted fitted pipeline (StandardScaler + KNN)
│   └── evaluation_results.json # Serialized evaluation metrics & metadata
├── static/
│   ├── style.css               # Clean, light-themed responsive CSS
│   └── plots/
│       ├── species_distribution.png   # Class balance bar chart
│       ├── sepal_scatter.png          # Sepal Length vs Width scatter plot
│       ├── petal_scatter.png          # Petal Length vs Width scatter plot
│       ├── feature_distributions.png  # Boxplots of all 4 features
│       ├── correlation_heatmap.png    # Pearson correlation matrix
│       └── confusion_matrix.png       # Test set confusion matrix
└── templates/
    └── index.html              # Modern light-themed dashboard with prediction form
```

---

## 7. Installation & Execution Guide

### Step 1: Install Dependencies
Open terminal or command prompt in the project root:
```bash
pip install -r requirements.txt
```

### Step 2: Run Model Training & Generate Artifacts
Run the standalone training script:
```bash
python train_model.py
```
This script:
1. Loads Scikit-learn's Iris dataset.
2. Creates and saves `data/iris.csv`.
3. Performs data quality and preprocessing checks.
4. Generates all 5 EDA plots and saves them in `static/plots/`.
5. Executes stratified 80/20 train-test split.
6. Performs 5-fold cross-validation on training data to verify $K=5$.
7. Trains the `StandardScaler + KNeighborsClassifier` pipeline.
8. Evaluates on the held-out test set and saves `static/plots/confusion_matrix.png`.
9. Persists `models/iris_knn_model.pkl` and `models/evaluation_results.json`.

### Step 3: Run the Flask Web Application
Start the Flask server:
```bash
python app.py
```
Access the application in your browser at:
```
http://127.0.0.1:5000/
```

---

## 8. How to Use the Prediction Form

1. Navigate to the **Iris Species Prediction** section at the bottom of the page (or click **"Predict Flower"** in the top navigation).
2. You can either:
   - Click one of the **Demo Archetype buttons** (Setosa, Versicolor, or Virginica) to auto-fill representative measurements.
   - Or manually type in continuous measurements:
     - **Sepal Length:** e.g., `5.1` cm
     - **Sepal Width:** e.g., `3.5` cm
     - **Petal Length:** e.g., `1.4` cm
     - **Petal Width:** e.g., `0.2` cm
3. Click **🌸 Predict Species**.
4. The application displays:
   - The predicted species name (*Setosa*, *Versicolor*, or *Virginica*).
   - Model-estimated class probabilities (e.g. Setosa: 100.0%, Versicolor: 0.0%, Virginica: 0.0%).
   - Educational disclaimer note.

---

## 9. Laboratory Record Figure Guide

The web application is designed for capturing the 6 figures required for the laboratory record:

- **Fig. 1 — Dataset Collection:** Dataset preview table (first 5 records), sample counts, and feature dimensions.
- **Fig. 2 — Data Preprocessing:** Missing values summary (0 found), duplicate check, data types, and target separation.
- **Fig. 3 — Exploratory Data Analysis (EDA):** The 5 actual Matplotlib/Seaborn plots (Species distribution, Sepal scatter, Petal scatter, Feature distributions, Correlation heatmap).
- **Fig. 4 — Model Training:** Train-test split (120/30), feature scaling with StandardScaler, K-value cross-validation table ($K=3, 5, 7, 9$), and fitted pipeline.
- **Fig. 5 — Model Evaluation:** Test accuracy, macro precision, recall, F1-score, confusion matrix plot, and per-class classification report.
- **Fig. 6 — Iris Species Prediction:** Flower measurement form, predicted species badge, and class probability distribution.
