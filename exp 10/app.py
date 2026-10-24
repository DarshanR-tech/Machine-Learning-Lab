"""
Iris Flower Classification Using K-Nearest Neighbors (KNN)
Flask Web Application for College ML Laboratory Demonstration
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
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(MODELS_DIR, 'iris_knn_model.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'evaluation_results.json')

app = Flask(__name__)

# Ensure model and metadata exist; if not, train once
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    print("[*] Model or metadata missing. Running training script now...")
    from train_model import run_pipeline
    run_pipeline()

# Load saved pipeline and evaluation metadata
print(f"[*] Loading fitted KNN pipeline from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

print(f"[*] Loading evaluation results from: {METADATA_PATH}")
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    evaluation_metadata = json.load(f)

TARGET_CLASSES = ['Setosa', 'Versicolor', 'Virginica']


def validate_input(val_str, field_name, min_val=0.1, max_val=15.0):
    if val_str is None or str(val_str).strip() == "":
        return None, f"Please enter a value for {field_name}."
    try:
        val = float(val_str)
        if np.isnan(val) or np.isinf(val):
            return None, f"{field_name} must be a valid finite number."
        if val < min_val or val > max_val:
            return None, f"{field_name} must be between {min_val} cm and {max_val} cm."
        return val, None
    except ValueError:
        return None, f"{field_name} must be a numerical value."


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    form_error = None
    input_values = {
        'sepal_length': '',
        'sepal_width': '',
        'petal_length': '',
        'petal_width': ''
    }

    if request.method == 'POST':
        # Retrieve form data
        raw_sl = request.form.get('sepal_length')
        raw_sw = request.form.get('sepal_width')
        raw_pl = request.form.get('petal_length')
        raw_pw = request.form.get('petal_width')

        input_values['sepal_length'] = raw_sl or ''
        input_values['sepal_width'] = raw_sw or ''
        input_values['petal_length'] = raw_pl or ''
        input_values['petal_width'] = raw_pw or ''

        # Validation
        sl, err1 = validate_input(raw_sl, "Sepal Length", 1.0, 10.0)
        sw, err2 = validate_input(raw_sw, "Sepal Width", 1.0, 6.0)
        pl, err3 = validate_input(raw_pl, "Petal Length", 0.5, 9.0)
        pw, err4 = validate_input(raw_pw, "Petal Width", 0.1, 4.0)

        error_list = [e for e in [err1, err2, err3, err4] if e is not None]

        if error_list:
            form_error = " • ".join(error_list)
        else:
            try:
                features = np.array([[sl, sw, pl, pw]])
                pred_idx = int(pipeline.predict(features)[0])
                probs = pipeline.predict_proba(features)[0]

                predicted_species = TARGET_CLASSES[pred_idx]
                probabilities = {
                    'Setosa': round(float(probs[0]) * 100, 2),
                    'Versicolor': round(float(probs[1]) * 100, 2),
                    'Virginica': round(float(probs[2]) * 100, 2)
                }

                prediction_result = {
                    'predicted_species': predicted_species,
                    'confidence': probabilities[predicted_species],
                    'probabilities': probabilities,
                    'inputs': {
                        'sepal_length': sl,
                        'sepal_width': sw,
                        'petal_length': pl,
                        'petal_width': pw
                    }
                }
            except Exception as e:
                form_error = f"Prediction failed: {str(e)}"

    return render_template(
        'index.html',
        metadata=evaluation_metadata,
        prediction=prediction_result,
        form_error=form_error,
        inputs=input_values
    )


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """AJAX endpoint for seamless live prediction without page refresh"""
    data = request.get_json() or {}
    raw_sl = data.get('sepal_length')
    raw_sw = data.get('sepal_width')
    raw_pl = data.get('petal_length')
    raw_pw = data.get('petal_width')

    sl, err1 = validate_input(raw_sl, "Sepal Length", 1.0, 10.0)
    sw, err2 = validate_input(raw_sw, "Sepal Width", 1.0, 6.0)
    pl, err3 = validate_input(raw_pl, "Petal Length", 0.5, 9.0)
    pw, err4 = validate_input(raw_pw, "Petal Width", 0.1, 4.0)

    error_list = [e for e in [err1, err2, err3, err4] if e is not None]
    if error_list:
        return jsonify({'success': False, 'error': " • ".join(error_list)}), 400

    try:
        features = np.array([[sl, sw, pl, pw]])
        pred_idx = int(pipeline.predict(features)[0])
        probs = pipeline.predict_proba(features)[0]

        predicted_species = TARGET_CLASSES[pred_idx]
        probabilities = {
            'Setosa': round(float(probs[0]) * 100, 2),
            'Versicolor': round(float(probs[1]) * 100, 2),
            'Virginica': round(float(probs[2]) * 100, 2)
        }

        return jsonify({
            'success': True,
            'predicted_species': predicted_species,
            'confidence': probabilities[predicted_species],
            'probabilities': probabilities,
            'inputs': {'sepal_length': sl, 'sepal_width': sw, 'petal_length': pl, 'petal_width': pw}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download/iris.csv')
def download_csv():
    return send_from_directory(DATA_DIR, 'iris.csv', as_attachment=True)


if __name__ == '__main__':
    port = 5000
    print("=" * 65)
    print("  🌸 FLASK APPLICATION STARTED")
    print(f"  Access local app at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host='127.0.0.1', port=port, debug=True)
