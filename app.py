"""
Coronary Heart Disease Prediction - Flask Web Application

This application uses machine learning to predict the 10-year risk of coronary heart disease
based on the Framingham Heart Study dataset.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import json
import os

from utils import load_model, load_scaler, FEATURE_ORDER, prepare_sample

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-chd-app'
# Load the trained model and scaler using shared utils
model = load_model()
scaler = load_scaler()

# User storage setup
USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {
            'admin': {'password': 'admin', 'role': 'admin'}
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f, indent=4)
        return default_users
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {'admin': {'password': 'admin', 'role': 'admin'}}

def save_users(users_dict):
    with open(USERS_FILE, 'w') as f:
        json.dump(users_dict, f, indent=4)

# Load existing users or create default admin
USERS = load_users()


@app.route('/')
def home():
    """Home page with prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - accepts patient data and returns risk assessment."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username]['password'] == password:
            session['logged_in'] = True
            session['username'] = username
            session['role'] = USERS[username]['role']
            if session['role'] == 'admin':
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS:
            return render_template('signup.html', error='Username already exists')
        USERS[username] = {'password': password, 'role': 'user'}
        save_users(USERS)
        session['logged_in'] = True
        session['username'] = username
        session['role'] = 'user'
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/admin')
def admin():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    # Load dataset stats for accurate data representation
    try:
        df = pd.read_csv('framingham.csv')
        stats = {
            'total_records': len(df),
            'high_risk': len(df[df['TenYearCHD'] == 1]),
            'low_risk': len(df[df['TenYearCHD'] == 0]),
            'avg_age': round(df['age'].mean(), 1),
            'avg_chol': round(df['totChol'].mean(), 1),
            'hypertension_percent': round((df['prevalentHyp'].sum() / len(df)) * 100, 1)
        }
    except Exception as e:
        stats = {
            'total_records': 0, 'high_risk': 0, 'low_risk': 0, 
            'avg_age': 0, 'avg_chol': 0, 'hypertension_percent': 0
        }
    return render_template('admin.html', stats=stats, users=USERS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized, please login first'}), 401
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
