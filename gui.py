"""
Basic GUI for Coronary Heart Disease Prediction
Uses multiple ML models for comparison and prediction
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
from tkinter import *
import tkinter.messagebox as M
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ---------- Data Loading & Preprocessing ----------
def load_and_preprocess_data():
    """Load and preprocess the Framingham dataset"""
    # Try to find the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = [
        os.path.join(script_dir, "framingham.csv"),
        "framingham.csv",
        r"H:\mldataset\framingham.csv"
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Could not find framingham.csv")
    
    # Drop education column
    df1 = df.drop(columns=["education"])
    
    # Drop features
    features_to_drop = ['currentSmoker', 'diaBP']
    df2 = df1.drop(columns=features_to_drop)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df3 = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns, index=df2.index)
    
    # Remove outliers
    df3 = df3[~(df3['sysBP'] > 220)]
    df3 = df3[~(df3['BMI'] > 43)]
    df3 = df3[~(df3['heartRate'] > 125)]
    df3 = df3[~(df3['glucose'] > 200)]
    df3 = df3[~(df3['totChol'] > 450)]
    
    return df3

def train_models(df3):
    """Train all ML models and return them"""
    # Standardize features
    scaler = StandardScaler()
    cols_to_standardise = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
    df3[cols_to_standardise] = scaler.fit_transform(df3[cols_to_standardise])
    
    # Split features and target
    X = df3.drop(columns=["TenYearCHD"])
    Y = df3["TenYearCHD"]
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
    
    models = {}
    accuracies = {}
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    models['KNN'] = knn
    accuracies['KNN'] = round(accuracy_score(Y_test, knn.predict(X_test)) * 100, 2)
    
    # Logistic Regression
    lg = LogisticRegression(max_iter=1000)
    lg.fit(X_train, Y_train)
    models['Logistic Regression'] = lg
    accuracies['Logistic Regression'] = round(accuracy_score(Y_test, lg.predict(X_test)) * 100, 2)
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    models['Naive Bayes'] = nb
    accuracies['Naive Bayes'] = round(accuracy_score(Y_test, nb.predict(X_test)) * 100, 2)
    
    # Decision Tree
    dt = DecisionTreeClassifier(min_samples_split=50, random_state=0)
    dt.fit(X_train, Y_train)
    models['Decision Tree'] = dt
    accuracies['Decision Tree'] = round(accuracy_score(Y_test, dt.predict(X_test)) * 100, 2)
    
    # SVM
    svc = SVC(C=1, kernel='rbf')
    svc.fit(X_train, Y_train)
    models['SVM'] = svc
    accuracies['SVM'] = round(accuracy_score(Y_test, svc.predict(X_test)) * 100, 2)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, Y_train)
    models['Random Forest'] = rf
    accuracies['Random Forest'] = round(accuracy_score(Y_test, rf.predict(X_test)) * 100, 2)
    
    return models, accuracies, scaler, list(X.columns)

# Load data and train models
print("Loading data and training models...")
df3 = load_and_preprocess_data()
models, accuracies, scaler, feature_order = train_models(df3)
print("Models trained successfully!")

# Get best model
best_model_name = max(accuracies.items(), key=lambda x: x[1])[0]

# ---------- GUI ----------
class CHDApp:
    def __init__(self):
        self.window = Tk()
        self.window.configure(bg='gray')
        self.window.title("CORONARY HEART DISEASE PREDICTION")
        
        # String variables
        self.v1 = StringVar()
        self.v2 = StringVar()
        self.v3 = StringVar()
        self.v4 = StringVar()
        self.v5 = StringVar()
        self.v6 = StringVar()
        self.v7 = StringVar()
        self.v8 = StringVar()
        self.v9 = StringVar()
        self.v10 = StringVar()
        self.v11 = StringVar()
        self.v12 = StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        L0 = Label(self.window, relief="solid", font=('arial', 20, 'bold'),
                   text="CORONARY HEART DISEASE PREDICTION USING MACHINE LEARNING",
                   bg='white', fg='red')
        L0.grid(row=1, column=1, columnspan=4)
        
        # Model buttons
        B1 = Button(self.window, pady=5, bd=5, bg="violet", command=self.knn,
                    text='KNearest Neighbors', font=('arial', 12, 'bold'))
        B2 = Button(self.window, pady=5, bd=5, bg="violet", command=self.logreg,
                    text='Logistic Regression', font=('arial', 12, 'bold'))
        B3 = Button(self.window, pady=5, bd=5, bg="violet", command=self.naivebayes,
                    text='Naive Bayes', font=('arial', 12, 'bold'))
        B4 = Button(self.window, pady=5, bd=5, bg="violet", command=self.decisiontree,
                    text='Decision Tree', font=('arial', 12, 'bold'))
        B5 = Button(self.window, pady=5, bd=5, bg="violet", command=self.svm,
                    text='Support Vector Machine', font=('arial', 12, 'bold'))
        B6 = Button(self.window, pady=5, bd=5, bg="violet", command=self.rf,
                    text='Random Forest', font=('arial', 12, 'bold'))
        
        B1.grid(row=2, column=1)
        B2.grid(row=2, column=2)
        B3.grid(row=2, column=3)
        B4.grid(row=2, column=4)
        B5.grid(row=3, column=1)
        B6.grid(row=3, column=2)
        
        # Model comparison button
        Bcmp = Button(self.window, bd=5, pady=8, relief="solid", bg='white', fg="blue",
                      command=self.compare, text='MODEL COMPARISON', font=('arial', 15, 'bold'))
        Bcmp.grid(row=3, column=3, columnspan=2)
        
        # Input labels and entries
        labels = [
            ('MALE (1/0)', self.v1),
            ('AGE', self.v2),
            ('CIGARETTES PER DAY', self.v3),
            ('BP MEDICINE (1/0)', self.v4),
            ('HEART STROKE IN PAST (1/0)', self.v5),
            ('HYPERTENSION (1/0)', self.v6),
            ('DIABETES (1/0)', self.v7),
            ('CHOLESTEROL', self.v8),
            ('SYS BP', self.v9),
            ('BODY MASS INDEX', self.v10),
            ('HEART RATE', self.v11),
            ('GLUCOSE', self.v12)
        ]
        
        for i, (text, var) in enumerate(labels):
            row = 4 + (i // 2)
            col = 1 + (i % 2) * 2
            
            Label(self.window, text=text, font=('arial', 12, 'bold')).grid(row=row, column=col, sticky='w')
            Entry(self.window, bd=5, textvariable=var, bg='cyan', font=('arial', 12, 'bold')).grid(row=row, column=col + 1)
        
        # Submit and Clear buttons
        Bsub = Button(self.window, bd=5, relief="solid", bg='white', fg="blue",
                      text="SUBMIT", font=('arial', 15, 'bold'), command=self.predict)
        Bsub.grid(row=10, column=3)
        
        Bres = Button(self.window, bd=5, relief="solid", bg='white', fg='blue',
                      text="CLEAR DATA", font=('arial', 15, 'bold'), command=self.reset)
        Bres.grid(row=10, column=4)
        
        # Footer
        Label(self.window, relief="solid", font=('arial', 20, 'bold'),
              text="CORONARY HEART DISEASE PREDICTION USING MACHINE LEARNING",
              bg='white', fg='red').grid(row=11, column=1, columnspan=4)
    
    def knn(self):
        M.showinfo(title="KNearest Neighbors", message=f"Accuracy: {accuracies['KNN']}%")
    
    def logreg(self):
        M.showinfo(title="Logistic Regression", message=f"Accuracy: {accuracies['Logistic Regression']}%")
    
    def naivebayes(self):
        M.showinfo(title="Naive Bayes", message=f"Accuracy: {accuracies['Naive Bayes']}%")
    
    def decisiontree(self):
        M.showinfo(title="Decision Tree", message=f"Accuracy: {accuracies['Decision Tree']}%")
    
    def svm(self):
        M.showinfo(title="SVM", message=f"Accuracy: {accuracies['SVM']}%")
    
    def rf(self):
        M.showinfo(title="Random Forest", message=f"Accuracy: {accuracies['Random Forest']}%")
    
    def compare(self):
        try:
            import matplotlib.pyplot as plt
            models_list = list(accuracies.keys())
            acc_list = list(accuracies.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models_list, acc_list, color=["green", "blue", "yellow", "red", "cyan", "magenta"],
                          edgecolor="black", linewidth=2)
            plt.xlabel("MODELS", fontsize=14, color="red")
            plt.ylabel("ACCURACY (%)", fontsize=14, color="blue")
            plt.title("Model Comparison", fontsize=16)
            plt.ylim(0, 100)
            
            for bar, acc in zip(bars, acc_list):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{acc}%', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            M.showinfo(title='BEST MODEL', message=f"Best Model: {best_model_name}\nAccuracy: {accuracies[best_model_name]}%")
        except ImportError:
            M.showerror("Error", "matplotlib is required for model comparison.\nInstall it with: pip install matplotlib")
    
    def predict(self):
        try:
            # Get input values
            inputs = {
                'male': float(self.v1.get()),
                'age': float(self.v2.get()),
                'cigsPerDay': float(self.v3.get()),
                'BPMeds': float(self.v4.get()),
                'prevalentStroke': float(self.v5.get()),
                'prevalentHyp': float(self.v6.get()),
                'diabetes': float(self.v7.get()),
                'totChol': float(self.v8.get()),
                'sysBP': float(self.v9.get()),
                'BMI': float(self.v10.get()),
                'heartRate': float(self.v11.get()),
                'glucose': float(self.v12.get())
            }
            
            # Create feature array in correct order
            feature_values = [inputs[f] for f in feature_order]
            
            # Scale the features that need scaling
            sample = pd.DataFrame([feature_values], columns=feature_order)
            cols_to_scale = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
            sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])
            
            # Predict using best model
            result = models[best_model_name].predict(sample.values)[0]
            
            if result == 1:
                M.showinfo(title="Heart Disease Prediction",
                          message="⚠️ HIGH RISK: You may have increased risk of coronary heart disease.\n\nPlease consult a healthcare professional.")
            else:
                M.showinfo(title="Heart Disease Prediction",
                          message="✅ LOW RISK: Your risk appears to be lower.\n\nContinue maintaining a healthy lifestyle!")
                
        except ValueError as e:
            M.showerror("Input Error", f"Please enter valid numeric values.\n\nError: {str(e)}")
        except Exception as e:
            M.showerror("Error", f"An error occurred: {str(e)}")
    
    def reset(self):
        self.v1.set("")
        self.v2.set("")
        self.v3.set("")
        self.v4.set("")
        self.v5.set("")
        self.v6.set("")
        self.v7.set("")
        self.v8.set("")
        self.v9.set("")
        self.v10.set("")
        self.v11.set("")
        self.v12.set("")
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CHDApp()
    app.run()