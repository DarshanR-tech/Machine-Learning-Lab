"""
Iris Flower Classification Using K-Nearest Neighbors (KNN)
Complete Machine Learning Training and Workflow Script
Suitable for College ML Laboratory Demonstration
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_pipeline():
    print("=" * 70)
    print("  🌸 IRIS FLOWER CLASSIFICATION USING K-NEAREST NEIGHBORS (KNN)")
    print("  Machine Learning Workflow & Laboratory Demonstration")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: Problem Definition
    # ---------------------------------------------------------
    print("\n[Step 1] Problem Definition:")
    print("  Predict the species of an Iris flower (Setosa, Versicolor, Virginica)")
    print("  based on four botanical morphology measurements: Sepal Length,")
    print("  Sepal Width, Petal Length, and Petal Width using K-Nearest Neighbors.")

    # ---------------------------------------------------------
    # STEP 2: Data Collection
    # ---------------------------------------------------------
    print("\n[Step 2] Data Collection:")
    raw_iris = load_iris()
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_names = list(raw_iris.target_names)  # ['setosa', 'versicolor', 'virginica']

    # Create DataFrame
    df = pd.DataFrame(raw_iris.data, columns=feature_names)
    df['target'] = raw_iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    # Save to data/iris.csv
    csv_path = os.path.join(DATA_DIR, 'iris.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Loaded built-in Scikit-learn Iris dataset: {df.shape[0]} samples, {len(feature_names)} features")
    print(f"  ✓ Saved copy of dataset to: {csv_path}")
    print(f"  ✓ Target Classes: {target_names}")

    # First five records
    first_five = df.head().to_dict(orient='records')
    class_distribution = df['species'].value_counts().to_dict()
    print(f"  ✓ Class Distribution: {class_distribution}")

    # ---------------------------------------------------------
    # STEP 3: Data Preprocessing
    # ---------------------------------------------------------
    print("\n[Step 3] Data Preprocessing:")
    missing_values = int(df[feature_names].isnull().sum().sum())
    missing_per_col = df[feature_names].isnull().sum().to_dict()
    duplicate_count = int(df[feature_names].duplicated().sum())
    data_types = {col: str(df[col].dtype) for col in df.columns}
    all_numerical = all(np.issubdtype(df[col].dtype, np.number) for col in feature_names)

    print(f"  ✓ Missing Values: {missing_values}")
    print(f"  ✓ Duplicate Feature Rows: {duplicate_count} (valid biological identical samples preserved)")
    print(f"  ✓ Numerical Features Verified: {all_numerical}")
    print(f"  ✓ Column Data Types: {data_types}")

    # ---------------------------------------------------------
    # STEP 4: Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    print("\n[Step 4] Exploratory Data Analysis (Generating Plots):")
    # Set clean light aesthetic for plots
    sns.set_theme(style="whitegrid", palette="muted")
    palette = {'setosa': '#4f46e5', 'versicolor': '#059669', 'virginica': '#db2777'}

    # 1. Species Distribution
    plt.figure(figsize=(6, 4.5))
    ax = sns.countplot(x='species', data=df, palette=palette, hue='species', legend=False)
    plt.title('Species Distribution in Iris Dataset', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Flower Species', fontsize=11)
    plt.ylabel('Count (Samples)', fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    plt.tight_layout()
    plot_dist_path = os.path.join(PLOTS_DIR, 'species_distribution.png')
    plt.savefig(plot_dist_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/species_distribution.png")

    # 2. Sepal Length vs Sepal Width
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette=palette, s=70, alpha=0.9, edgecolor='w')
    plt.title('Sepal Length vs Sepal Width', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Sepal Length (cm)', fontsize=11)
    plt.ylabel('Sepal Width (cm)', fontsize=11)
    plt.legend(title='Species', frameon=True)
    plt.tight_layout()
    plot_sepal_path = os.path.join(PLOTS_DIR, 'sepal_scatter.png')
    plt.savefig(plot_sepal_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/sepal_scatter.png")

    # 3. Petal Length vs Petal Width
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette=palette, s=70, alpha=0.9, edgecolor='w')
    plt.title('Petal Length vs Petal Width (Strong Separability)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Petal Length (cm)', fontsize=11)
    plt.ylabel('Petal Width (cm)', fontsize=11)
    plt.legend(title='Species', frameon=True)
    plt.tight_layout()
    plot_petal_path = os.path.join(PLOTS_DIR, 'petal_scatter.png')
    plt.savefig(plot_petal_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/petal_scatter.png")

    # 4. Feature Distributions (Boxplots for all 4 features)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    titles = [
        ('sepal_length', 'Sepal Length (cm)'),
        ('sepal_width', 'Sepal Width (cm)'),
        ('petal_length', 'Petal Length (cm)'),
        ('petal_width', 'Petal Width (cm)')
    ]
    for idx, (col, lbl) in enumerate(titles):
        ax_i = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x='species', y=col, palette=palette, ax=ax_i, hue='species', legend=False)
        ax_i.set_title(lbl, fontsize=11, fontweight='bold')
        ax_i.set_xlabel('')
        ax_i.set_ylabel('cm', fontsize=10)
    plt.suptitle('Measurement Distributions Across Iris Species', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_feat_path = os.path.join(PLOTS_DIR, 'feature_distributions.png')
    plt.savefig(plot_feat_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/feature_distributions.png")

    # 5. Correlation Heatmap
    plt.figure(figsize=(6, 5))
    corr = df[feature_names].corr()
    corr.columns = ['Sepal L.', 'Sepal W.', 'Petal L.', 'Petal W.']
    corr.index = ['Sepal L.', 'Sepal W.', 'Petal L.', 'Petal W.']
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True, vmin=-1, vmax=1, linewidths=1)
    plt.title('Feature Correlation Heatmap (Pearson r)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plot_corr_path = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(plot_corr_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/correlation_heatmap.png")

    # ---------------------------------------------------------
    # STEP 5: Feature Selection
    # ---------------------------------------------------------
    print("\n[Step 5] Feature Selection:")
    print("  Selected Features: Sepal Length, Sepal Width, Petal Length, Petal Width.")
    print("  Rationale: All 4 morphological features represent critical botanical dimensions.")
    print("  Petal measurements provide distinct clustering, while Sepals distinguish boundary specimens.")

    # ---------------------------------------------------------
    # STEP 6: Train-Test Split
    # ---------------------------------------------------------
    print("\n[Step 6] Train-Test Split (80% Train, 20% Test):")
    X = df[feature_names].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Training Samples: {len(X_train)} (40 Setosa, 40 Versicolor, 40 Virginica)")
    print(f"  ✓ Testing Samples:  {len(X_test)} (10 Setosa, 10 Versicolor, 10 Virginica)")

    # ---------------------------------------------------------
    # STEP 7 & 8: Feature Scaling & KNN Model Training with Pipeline
    # ---------------------------------------------------------
    print("\n[Step 7 & 8] Feature Scaling & KNN Pipeline Training:")
    # Evaluate K values using 5-fold cross-validation on TRAINING set only (no test set leakage!)
    k_options = [3, 5, 7, 9]
    cv_results = {}
    best_k = 5
    best_score = 0.0

    for k_val in k_options:
        temp_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k_val, weights='uniform', metric='euclidean'))
        ])
        scores = cross_val_score(temp_pipe, X_train, y_train, cv=5, scoring='accuracy')
        mean_acc = float(scores.mean())
        std_acc = float(scores.std())
        cv_results[k_val] = {'mean_accuracy': round(mean_acc * 100, 2), 'std': round(std_acc * 100, 2)}
        print(f"    • K={k_val}: 5-Fold CV Accuracy = {mean_acc * 100:.2f}% (±{std_acc * 100:.2f}%)")
        if mean_acc > best_score:
            best_score = mean_acc
            best_k = k_val

    print(f"  ✓ Selected K={best_k} for final model based on training cross-validation.")

    # Build final pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='uniform', metric='euclidean'))
    ])

    # Fit pipeline on training data ONLY
    pipeline.fit(X_train, y_train)

    # Save pipeline
    model_path = os.path.join(MODELS_DIR, 'iris_knn_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"  ✓ Saved complete fitted pipeline (StandardScaler + KNN) to: {model_path}")

    # ---------------------------------------------------------
    # STEP 9: Model Evaluation on Held-Out Test Set
    # ---------------------------------------------------------
    print("\n[Step 9] Model Evaluation on Held-Out Test Set:")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average='macro'))
    rec_macro = float(recall_score(y_test, y_pred, average='macro'))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))

    print(f"  ✓ Test Set Accuracy:  {acc * 100:.2f}%")
    print(f"  ✓ Macro Precision:    {prec_macro * 100:.2f}%")
    print(f"  ✓ Macro Recall:       {rec_macro * 100:.2f}%")
    print(f"  ✓ Macro F1-Score:     {f1_macro * 100:.2f}%")

    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print("\n  Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"    {target_names[i]:10}: {row}")

    # Plot Confusion Matrix
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    plt.title(f'Confusion Matrix (Test Set, K={best_k})', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Species', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Species', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plot_cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(plot_cm_path, dpi=180)
    plt.close()
    print(f"  ✓ Saved: static/plots/confusion_matrix.png")

    # ---------------------------------------------------------
    # STEP 10: Persist Complete Workflow Metadata for Web Display
    # ---------------------------------------------------------
    summary_data = {
        "project_name": "Iris Flower Classification Using K-Nearest Neighbors (KNN)",
        "dataset": {
            "total_samples": int(len(df)),
            "num_features": len(feature_names),
            "features": ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"],
            "classes": ["Setosa", "Versicolor", "Virginica"],
            "class_distribution": class_distribution,
            "first_five_records": first_five,
            "shape": list(df.shape)
        },
        "preprocessing": {
            "missing_values": missing_values,
            "missing_per_col": missing_per_col,
            "duplicates": duplicate_count,
            "data_types": data_types,
            "all_numerical": all_numerical
        },
        "training": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_ratio": "80% Training / 20% Testing (Stratified)",
            "selected_k": best_k,
            "cv_comparison": cv_results,
            "algorithm": "K-Nearest Neighbors (KNN)",
            "metric": "Euclidean (L2 distance)",
            "scaling": "StandardScaler (fitted on training data only)"
        },
        "evaluation": {
            "accuracy": round(acc * 100, 2),
            "precision_macro": round(prec_macro * 100, 2),
            "recall_macro": round(rec_macro * 100, 2),
            "f1_macro": round(f1_macro * 100, 2),
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                sp: {
                    "precision": round(report_dict[sp]['precision'] * 100, 2),
                    "recall": round(report_dict[sp]['recall'] * 100, 2),
                    "f1_score": round(report_dict[sp]['f1-score'] * 100, 2),
                    "support": int(report_dict[sp]['support'])
                }
                for sp in target_names
            }
        }
    }

    results_json_path = os.path.join(MODELS_DIR, 'evaluation_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✓ Saved evaluation summary metadata to: {results_json_path}")
    print("\n" + "=" * 70)
    print("  🎉 TRAINING & WORKFLOW ARTIFACTS COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline()
