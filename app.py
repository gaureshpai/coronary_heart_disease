"""
Coronary Heart Disease Prediction - Flask Web Application

This application uses machine learning to predict the 10-year risk of coronary heart disease
based on the Framingham Heart Study dataset.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd

from utils import load_model, load_scaler, FEATURE_ORDER, prepare_sample

app = Flask(__name__)

# Load the trained model and scaler using shared utils
model = load_model()
scaler = load_scaler()


@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - accepts patient data and returns risk assessment."""
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()

            input_data = {
                'male': int(form_data['male']),
                'age': float(form_data['age']),
                'cigsPerDay': float(form_data['cigsPerDay']),
                'BPMeds': int(form_data['BPMeds']),
                'prevalentStroke': int(form_data['prevalentStroke']),
                'prevalentHyp': int(form_data['prevalentHyp']),
                'diabetes': int(form_data['diabetes']),
                'totChol': float(form_data['totChol']),
                'sysBP': float(form_data['sysBP']),
                'BMI': float(form_data['BMI']),
                'heartRate': float(form_data['heartRate']),
                'glucose': float(form_data['glucose'])
            }

            sample = prepare_sample(input_data, scaler)

            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1]

            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if prediction == 1 else 'Low',
                'message': ('High risk of coronary heart disease' if prediction == 1
                            else 'Low risk of coronary heart disease')
            }

            return render_template('result.html', result=result)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)."""
    try:
        data = request.get_json()

        input_data = {col: data[col] for col in FEATURE_ORDER}
        sample = prepare_sample(input_data, scaler)

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'message': ('High risk of coronary heart disease' if prediction == 1
                        else 'Low risk of coronary heart disease')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
