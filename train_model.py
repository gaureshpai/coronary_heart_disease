"""
Train and save the coronary heart disease prediction model.

This script loads the Framingham dataset, preprocesses it, applies SMOTE to handle
class imbalance, trains multiple ML models with class_weight='balanced', and saves
the best model and scaler for use by the Flask application.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# NOTE: We use SMOTE to handle class imbalance. Since SMOTE already balances
# the training data, we do NOT also use class_weight='balanced' to avoid
# over-correction which can hurt performance.

# Create model directory if it doesn't exist
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'framingham.csv')


def load_and_preprocess():
    """Load and preprocess the Framingham dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Drop unnecessary columns
    df = df.drop(columns=["education", "currentSmoker", "diaBP"])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Remove outliers
    outlier_conditions = [
        df['sysBP'] > 220,
        df['BMI'] > 43,
        df['heartRate'] > 125,
        df['glucose'] > 200,
        df['totChol'] > 450
    ]
    for condition in outlier_conditions:
        df = df[~condition]
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    
    return df


def prepare_features(df):
    """Prepare features and target variable."""
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    # Standardize features
    scaler = StandardScaler()
    cols_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    
    return X, y, scaler


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance in training data."""
    print(f"\nClass distribution BEFORE SMOTE:")
    print(f"  No CHD (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    print(f"  CHD (1):    {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nClass distribution AFTER SMOTE:")
    print(f"  No CHD (0): {(y_resampled == 0).sum()} ({(y_resampled == 0).sum() / len(y_resampled) * 100:.1f}%)")
    print(f"  CHD (1):    {(y_resampled == 1).sum()} ({(y_resampled == 1).sum() / len(y_resampled) * 100:.1f}%)")
    print(f"  Total: {len(y_resampled)} samples (was {len(y_train)})")
    
    return X_resampled, y_resampled


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models with class_weight='balanced' and return their accuracies."""
    models = {}
    accuracies = {}
    
    # KNN (no class_weight support)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    accuracies['KNN'] = round(accuracy_score(y_test, knn.predict(X_test)) * 100, 2)
    
    # Logistic Regression
    lg = LogisticRegression(max_iter=1000, random_state=42)
    lg.fit(X_train, y_train)
    models['Logistic Regression'] = lg
    accuracies['Logistic Regression'] = round(accuracy_score(y_test, lg.predict(X_test)) * 100, 2)
    
    # Naive Bayes (no class_weight support)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    accuracies['Naive Bayes'] = round(accuracy_score(y_test, nb.predict(X_test)) * 100, 2)
    
    # Decision Tree
    dt = DecisionTreeClassifier(min_samples_split=50, random_state=0)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    accuracies['Decision Tree'] = round(accuracy_score(y_test, dt.predict(X_test)) * 100, 2)
    
    # SVM
    svc = SVC(C=1, kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train, y_train)
    models['SVM'] = svc
    accuracies['SVM'] = round(accuracy_score(y_test, svc.predict(X_test)) * 100, 2)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    accuracies['Random Forest'] = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 2)
    
    return models, accuracies


def main():
    """Main training pipeline."""
    # Load and preprocess data
    df = load_and_preprocess()
    
    # Prepare features
    X, y, scaler = prepare_features(df)
    
    # Train-test split (BEFORE SMOTE to keep test data realistic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Apply SMOTE to training data only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Train all models with balanced classes
    print("\n" + "="*50)
    print("TRAINING MODELS (with SMOTE + class_weight='balanced')")
    print("="*50)
    models, accuracies = train_all_models(X_train_resampled, X_test, y_train_resampled, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL ACCURACIES")
    print("="*50)
    for name, acc in accuracies.items():
        print(f"{name}: {acc}%")
    
    # Find best model
    best_model_name = max(accuracies.items(), key=lambda kv: kv[1])[0]
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} ({accuracies[best_model_name]}%)")
    
    # Save best model and scaler
    print("\nSaving model and scaler...")
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save all models for comparison
    with open(os.path.join(MODEL_DIR, 'all_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    # Save accuracies
    with open(os.path.join(MODEL_DIR, 'accuracies.pkl'), 'wb') as f:
        pickle.dump(accuracies, f)
    
    print(f"\nModel saved to: {MODEL_DIR}/model.pkl")
    print(f"Scaler saved to: {MODEL_DIR}/scaler.pkl")
    print("\nTraining complete!")
    
    # Print classification report for best model
    print(f"\n{'='*50}")
    print(f"CLASSIFICATION REPORT ({best_model_name})")
    print("="*50)
    print(classification_report(y_test, best_model.predict(X_test)))
    
    # Print confusion matrix
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")


if __name__ == '__main__':
    main()
