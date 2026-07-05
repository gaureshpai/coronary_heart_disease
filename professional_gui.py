"""
Professional GUI for Coronary Heart Disease Prediction
Features modern styling, model comparison, and probability outputs
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os

# ML imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox

# plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Data Loading & Preprocessing ----------
def load_and_preprocess_data():
    """Load and preprocess the Framingham dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = [
        os.path.join(script_dir, "framingham.csv"),
        "framingham.csv",
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
    y = df3["TenYearCHD"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    models = {}
    accuracies = {}
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    models['KNN'] = knn
    accuracies['KNN'] = round(accuracy_score(y_test, knn.predict(X_test)) * 100, 2)
    
    # Logistic Regression
    lg = LogisticRegression(max_iter=1000)
    lg.fit(X_train, y_train)
    models['Logistic Regression'] = lg
    accuracies['Logistic Regression'] = round(accuracy_score(y_test, lg.predict(X_test)) * 100, 2)
    
    # Naive Bayes
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
    svc = SVC(C=1, kernel='rbf', probability=True)
    svc.fit(X_train, y_train)
    models['SVM'] = svc
    accuracies['SVM'] = round(accuracy_score(y_test, svc.predict(X_test)) * 100, 2)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    accuracies['Random Forest'] = round(accuracy_score(y_test, rf.predict(X_test)) * 100, 2)
    
    return models, accuracies, scaler, list(X.columns)

# ---------- Build Sample ----------
def build_sample_from_inputs(inputs, scaler, feature_order):
    """Convert input dict to scaled numpy array"""
    cols_to_standardise = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']
    
    row = {}
    for col in feature_order:
        row[col] = float(inputs.get(col, 0))
    
    raw_df = pd.DataFrame([row])
    raw_df[cols_to_standardise] = scaler.transform(raw_df[cols_to_standardise])
    
    return raw_df[feature_order].values

# ---------- Main Application ----------
class CHDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Load and train models
        print("Loading data and training models...")
        df3 = load_and_preprocess_data()
        self.models, self.accuracies, self.scaler, self.feature_order = train_models(df3)
        self.best_model_name = max(self.accuracies.items(), key=lambda x: x[1])[0]
        print(f"Models trained! Best model: {self.best_model_name}")
        
        # Configure window
        self.title("Coronary Heart Disease Predictor")
        self.geometry("1000x650")
        self.resizable(False, False)
        self.configure(bg="#f4f7fb")
        
        self._create_widgets()
        self.status_var.set(f"Models trained. Best model: {self.best_model_name}")
    
    def _create_widgets(self):
        # Style configuration
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Green.TLabel', foreground='green', font=('Helvetica', 14, 'bold'))
        style.configure('Red.TLabel', foreground='red', font=('Helvetica', 14, 'bold'))
        
        # Header
        header = ttk.Label(self, text="Coronary Heart Disease Prediction Using ML", style='Title.TLabel')
        header.pack(pady=12)
        
        container = ttk.Frame(self)
        container.pack(fill="both", padx=12, pady=6, expand=True)
        
        # Left panel - Input
        left = ttk.LabelFrame(container, text="Patient Data (Enter numeric values)", padding=(12, 10))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        
        # Right panel - Models & Results
        right = ttk.LabelFrame(container, text="Models & Prediction", padding=(12, 10))
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        
        # Input fields configuration
        display_spec = [
            ("male", "Male (1=Yes, 0=No)", "1"),
            ("age", "Age (years)", "50"),
            ("cigsPerDay", "Cigarettes per day", "0"),
            ("BPMeds", "BP Medicine (1=Yes, 0=No)", "0"),
            ("prevalentStroke", "Prior Stroke (1=Yes, 0=No)", "0"),
            ("prevalentHyp", "Hypertension (1=Yes, 0=No)", "0"),
            ("diabetes", "Diabetes (1=Yes, 0=No)", "0"),
            ("totChol", "Total Cholesterol (mg/dL)", "200"),
            ("sysBP", "Systolic BP (mmHg)", "120"),
            ("BMI", "BMI", "25"),
            ("heartRate", "Heart Rate (bpm)", "70"),
            ("glucose", "Glucose (mg/dL)", "85"),
        ]
        
        self.inputs = {}
        for i, (col, label_text, default) in enumerate(display_spec):
            r = i // 2
            c = (i % 2) * 2
            
            lbl = ttk.Label(left, text=label_text)
            lbl.grid(row=r, column=c, sticky="w", padx=(0, 6), pady=6)
            
            ent_var = tk.StringVar(value=default)
            ent = ttk.Entry(left, textvariable=ent_var, width=18)
            ent.grid(row=r, column=c + 1, sticky="w", pady=6)
            self.inputs[col] = ent_var
        
        # Buttons frame
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=7, column=0, columnspan=4, pady=(12, 0))
        
        ttk.Button(btn_frame, text="Predict (Best Model)", command=self.predict_best).grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text="Predict (Choose Model)", command=self.predict_choose_model).grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text="Clear Fields", command=self.clear_fields).grid(row=0, column=2, padx=6)
        
        # Right panel - Model accuracy table
        ttk.Label(right, text="Model Accuracies (%)", style='Header.TLabel').pack(anchor="w")
        
        self.tree = ttk.Treeview(right, columns=("Model", "Accuracy"), show="headings", height=6)
        self.tree.heading("Model", text="Model")
        self.tree.heading("Accuracy", text="Accuracy (%)")
        self.tree.column("Model", width=200, anchor="center")
        self.tree.column("Accuracy", width=120, anchor="center")
        self.tree.pack(pady=6)
        
        # Populate table
        for name, acc in self.accuracies.items():
            tag = "best" if name == self.best_model_name else ""
            self.tree.insert("", "end", values=(name, acc), tags=(tag,))
        self.tree.tag_configure("best", background="#e6ffe6")
        
        # Model selection
        ttk.Label(right, text="Choose model for prediction:").pack(anchor="w", pady=(8, 2))
        self.model_choice = tk.StringVar(value=self.best_model_name)
        model_cb = ttk.Combobox(right, textvariable=self.model_choice,
                               values=list(self.models.keys()), state="readonly")
        model_cb.pack(anchor="w", pady=(0, 8))
        
        # Result display
        result_frame = ttk.Frame(right)
        result_frame.pack(fill="x", pady=(8, 0))
        
        ttk.Label(result_frame, text="Prediction Result:", style='Header.TLabel').grid(row=0, column=0, sticky="w")
        self.result_var = tk.StringVar(value="No prediction yet")
        self.result_label = ttk.Label(result_frame, textvariable=self.result_var)
        self.result_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        
        # Confidence display
        self.conf_var = tk.StringVar(value="")
        ttk.Label(result_frame, textvariable=self.conf_var).grid(row=2, column=0, sticky="w", pady=(6, 0))
        
        # Action buttons
        action_frame = ttk.Frame(right)
        action_frame.pack(fill="x", pady=(12, 0))
        
        ttk.Button(action_frame, text="Show Comparison Chart", command=self.show_chart).grid(row=0, column=0, padx=6)
        ttk.Button(action_frame, text="Export Accuracies CSV", command=self.export_accuracies).grid(row=0, column=1, padx=6)
        ttk.Button(action_frame, text="Exit", command=self.quit).grid(row=0, column=2, padx=6)
        
        # Status bar
        status = ttk.Frame(self)
        status.pack(fill="x", side="bottom", padx=6, pady=6)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x")
    
    def predict_best(self):
        self._predict_with_model_name(self.best_model_name)
    
    def predict_choose_model(self):
        model_name = self.model_choice.get()
        self._predict_with_model_name(model_name)
    
    def _predict_with_model_name(self, model_name):
        try:
            # Gather inputs
            inputs = {}
            for col in self.feature_order:
                if col in self.inputs:
                    inputs[col] = self.inputs[col].get()
                else:
                    inputs[col] = "0"
            
            # Build sample and predict
            sample = build_sample_from_inputs(inputs, self.scaler, self.feature_order)
            model = self.models[model_name]
            pred = model.predict(sample)[0]
            
            # Get probability if available
            prob_text = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample)[0]
                prob_text = f" | Probability (CHD): {proba[1] * 100:.2f}%"
            
            # Update result display
            if pred == 1:
                self.result_var.set("Prediction: HIGH RISK of CHD")
                self.result_label.configure(foreground="red")
            else:
                self.result_var.set("Prediction: LOW RISK of CHD")
                self.result_label.configure(foreground="green")
            
            self.conf_var.set(f"Model: {model_name} (Accuracy: {self.accuracies[model_name]}%){prob_text}")
            self.status_var.set("Prediction completed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")
            self.status_var.set("Error during prediction.")
    
    def clear_fields(self):
        for var in self.inputs.values():
            var.set("")
        self.status_var.set("Fields cleared.")
        self.result_var.set("No prediction yet")
        self.result_label.configure(foreground="black")
        self.conf_var.set("")
    
    def show_chart(self):
        try:
            names = list(self.accuracies.keys())
            vals = [self.accuracies[n] for n in names]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(names, vals)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Model Comparison (Accuracy)")
            
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}%", ha='center')
            
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display chart: {e}")
    
    def export_accuracies(self):
        try:
            out = pd.DataFrame(list(self.accuracies.items()), columns=["Model", "Accuracy"])
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_accuracies.csv")
            out.to_csv(out_path, index=False)
            messagebox.showinfo("Exported", f"Accuracies exported to:\n{out_path}")
            self.status_var.set("Accuracies exported.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self.status_var.set("Export failed.")
    
    def run(self):
        self.mainloop()

if __name__ == "__main__":
    app = CHDApp()
    app.run()