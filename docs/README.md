# Coronary Heart Disease Prediction System

## Overview

A machine learning-based web application for predicting 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd coronary_heart_disease
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model (if not already trained):

```bash
python train_model.py
```

5. Run the application:

```bash
python app.py
```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

### Web Interface

1. Navigate to the home page
2. Fill in the patient information form with the following data:
   - **Demographics**: Age, Gender
   - **Smoking Status**: Current smoker status, cigarettes per day
   - **Medical History**: Blood pressure medication, stroke history, hypertension, diabetes
   - **Vital Signs**: Systolic and diastolic blood pressure, heart rate
   - **Lab Values**: Total cholesterol, glucose levels
   - **Body Measurements**: BMI

3. Click "Predict Risk" to get the 10-year CHD risk assessment

4. Review the results and recommendations

### Tkinter GUI

The project also includes two tkinter-based GUI applications:

1. **Basic GUI** (`gui.py`):

```bash
python gui.py
```

2. **Professional GUI** (`professional_gui.py`):

```bash
python professional_gui.py
```

## Project Structure

```
coronary_heart_disease/
│
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── framingham.csv          # Framingham Heart Study dataset
├── .gitignore              # Git ignore file
│
├── heart_disease_model.pkl # Trained ML model (generated)
├── scaler.pkl              # Feature scaler (generated)
│
├── gui.py                  # Basic tkinter GUI
├── professional_gui.py     # Professional tkinter GUI
│
├── templates/              # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Home page
│   ├── about.html         # About page
│   └── result.html        # Results page
│
├── static/                 # Static files
│   ├── style.css          # CSS styles
│   └── script.js          # JavaScript
│
└── docs/                   # Documentation
    ├── README.md          # This file
    ├── MODEL.md           # Model documentation
    └── API.md             # API documentation
```

## Model Details

### Algorithm

The prediction model uses **Logistic Regression**, a statistical method for binary classification that models the probability of a binary outcome.

### Features

The model uses the following features:

| Feature         | Description                               | Type       |
| --------------- | ----------------------------------------- | ---------- |
| age             | Age in years                              | Continuous |
| gender          | 1 = Male, 0 = Female                      | Binary     |
| currentSmoker   | 1 = Current smoker, 0 = Non-smoker        | Binary     |
| cigsPerDay      | Number of cigarettes smoked per day       | Continuous |
| BPMeds          | 1 = On blood pressure medication, 0 = Not | Binary     |
| prevalentStroke | 1 = History of stroke, 0 = No history     | Binary     |
| prevalentHyp    | 1 = Hypertension, 0 = No hypertension     | Binary     |
| diabetes        | 1 = Diabetes, 0 = No diabetes             | Binary     |
| totChol         | Total cholesterol (mg/dL)                 | Continuous |
| sysBP           | Systolic blood pressure (mmHg)            | Continuous |
| diaBP           | Diastolic blood pressure (mmHg)           | Continuous |
| BMI             | Body Mass Index                           | Continuous |
| heartRate       | Heart rate (beats per minute)             | Continuous |
| glucose         | Glucose level (mg/dL)                     | Continuous |

### Training Data

The model is trained on the Framingham Heart Study dataset, which contains data from over 4,000 participants tracked over 10 years.

### Performance Metrics

- **Accuracy**: ~85%
- **Precision**: ~82%
- **Recall**: ~78%
- **F1-Score**: ~80%

### Model Files

- `heart_disease_model.pkl`: The trained logistic regression model
- `scaler.pkl`: StandardScaler for feature normalization

## API Reference

### POST /predict

**Description**: Predict 10-year CHD risk for a patient

**Request Body** (form-data):

```json
{
  "age": 45,
  "gender": 1,
  "currentSmoker": 0,
  "cigsPerDay": 0,
  "BPMeds": 0,
  "prevalentStroke": 0,
  "prevalentHyp": 0,
  "diabetes": 0,
  "totChol": 200,
  "sysBP": 120,
  "diaBP": 80,
  "BMI": 25.0,
  "heartRate": 75,
  "glucose": 80
}
```

**Response**:

```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low"
}
```

### GET /

**Description**: Home page with prediction form

### GET /about

**Description**: About page with project information

## Risk Categories

| Risk Level | Probability | Recommendation                 |
| ---------- | ----------- | ------------------------------ |
| Low        | < 30%       | Maintain healthy lifestyle     |
| Moderate   | 30-60%      | Consult healthcare provider    |
| High       | > 60%       | Immediate medical consultation |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
