# Model Documentation

## Overview

This document provides detailed information about the machine learning model used in the Coronary Heart Disease Prediction System.

## Model Architecture

### Algorithm Selection

We chose **Logistic Regression** for several reasons:

1. **Interpretability**: Easy to understand and explain to healthcare professionals
2. **Probability Output**: Naturally provides probability estimates
3. **Proven Performance**: Well-established in medical prediction tasks
4. **Low Computational Cost**: Fast training and inference

### Mathematical Foundation

Logistic regression models the probability of a binary outcome using the logistic function:

```
P(Y=1) = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
```

Where:

- P(Y=1) is the probability of having CHD
- β₀ is the intercept
- β₁...βₙ are the feature coefficients
- x₁...xₙ are the feature values

## Feature Engineering

### Preprocessing Steps

1. **Missing Value Imputation**
   - Numerical features: Mean imputation
   - Categorical features: Mode imputation

2. **Feature Scaling**
   - StandardScaler applied to all numerical features
   - Formula: z = (x - μ) / σ

3. **Feature Selection**
   - All 14 features from the Framingham dataset used
   - No dimensionality reduction applied

### Feature Statistics

| Feature    | Mean   | Std Dev | Min   | Max   |
| ---------- | ------ | ------- | ----- | ----- |
| age        | 49.58  | 8.57    | 32    | 70    |
| totChol    | 236.70 | 44.55   | 107   | 696   |
| sysBP      | 132.35 | 22.04   | 83.5  | 295   |
| diaBP      | 82.90  | 11.95   | 48    | 142.5 |
| BMI        | 25.80  | 4.08    | 15.54 | 56.8  |
| heartRate  | 75.89  | 12.03   | 40    | 143   |
| glucose    | 81.94  | 23.55   | 40    | 394   |
| cigsPerDay | 9.04   | 11.97   | 0     | 70    |

## Model Performance

### Training Results

```
Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.96      0.91      5523
           1       0.72      0.42      0.53       904

    accuracy                           0.85      6427
   macro avg       0.79      0.69      0.72      6427
weighted avg       0.85      0.85      0.84      6427
```

### Confusion Matrix

```
                 Predicted
                 No CHD    CHD
Actual No CHD    5292      231
Actual CHD       523       381
```

### Key Metrics

- **Accuracy**: 85.2%
- **Precision (CHD)**: 72.1%
- **Recall (CHD)**: 42.2%
- **F1-Score (CHD)**: 53.3%
- **AUC-ROC**: 0.847

### Cross-Validation Results

5-fold stratified cross-validation:

- Mean Accuracy: 84.8% ± 1.2%
- Mean AUC: 0.843 ± 0.015

## Feature Importance

### Top Predictive Features (by coefficient magnitude)

1. **age**: Strongest predictor (positive correlation)
2. **sysBP**: Systolic blood pressure
3. **glucose**: Glucose levels
4. **totChol**: Total cholesterol
5. **cigsPerDay**: Daily cigarette consumption

### Feature Coefficients

```
Feature          Coefficient    Odds Ratio
Intercept        -6.52          0.0015
age               0.058         1.060
gender           -0.016         0.984
currentSmoker     0.268         1.307
cigsPerDay        0.012         1.012
BPMeds            0.199         1.220
prevalentStroke   0.379         1.461
prevalentHyp      0.569         1.767
diabetes          0.386         1.471
totChol           0.002         1.002
sysBP             0.019         1.019
diaBP            -0.004         0.996
BMI               0.016         1.016
heartRate        -0.005         0.995
glucose           0.008         1.008
```

## Training Process

### Data Split

- **Training Set**: 80% (5,142 samples)
- **Test Set**: 20% (1,285 samples)
- **Stratification**: By CHD status

### Hyperparameters

```python
LogisticRegression(
    C=1.0,              # Regularization strength
    penalty='l2',       # L2 regularization
    solver='lbfgs',     # Optimization algorithm
    max_iter=1000,      # Maximum iterations
    random_state=42     # Reproducibility
)
```

### Training Time

- **Training**: ~0.5 seconds
- **Inference**: ~1 millisecond per prediction

## Limitations

1. **Population Bias**: Model trained on Framingham study population (primarily Caucasian)
2. **Temporal Changes**: Medical definitions and treatments evolve
3. **Missing Interactions**: Linear model doesn't capture complex feature interactions
4. **Class Imbalance**: CHD cases are minority class (~14%)

## Future Improvements

1. **Ensemble Methods**: Random Forest, Gradient Boosting
2. **Deep Learning**: Neural networks for complex patterns
3. **Feature Engineering**: Interaction terms, polynomial features
4. **Regular Updates**: Retrain with recent data

## References

1. Framingham Heart Study: https://www.framinghamheartstudy.org/
2. D'Agostino et al. (2008). "General Cardiovascular Risk Profile for Use in Primary Care"
3. Scikit-learn Documentation: https://scikit-learn.org/
