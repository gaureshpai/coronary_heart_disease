"""
Coronary Heart Disease Prediction - Flask Web Application

This application uses machine learning to predict the 10-year risk of coronary heart disease
based on the Framingham Heart Study dataset.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained model and scaler
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
model_path = os.path.join(MODEL_DIR, 'model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

# Load models if they exist
model = None
scaler = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    print("Warning: Model files not found. Please run train_model.py first.")


@app.route('/')
def home():
    """Home page with overview of the application."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - accepts patient data and returns risk assessment."""
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            
            # Convert to correct data types
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
            
            # Prepare data for prediction
            features = ['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                      'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'BMI',
                      'heartRate', 'glucose']
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data], columns=features)
            
            # Scale numerical features
            cols_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
            input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if prediction == 1 else 'Low',
                'message': 'High risk of coronary heart disease' if prediction == 1 
                          else 'Low risk of coronary heart disease'
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')


@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)."""
    try:
        data = request.get_json()
        
        features = ['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                    'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'BMI',
                    'heartRate', 'glucose']
        
        input_df = pd.DataFrame([data], columns=features)
        
        cols_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'message': 'High risk of coronary heart disease' if prediction == 1 
                      else 'Low risk of coronary heart disease'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
