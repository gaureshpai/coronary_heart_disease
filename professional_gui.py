"""
Professional GUI for Coronary Heart Disease Prediction
Loads pre-trained models from model/ directory.
Features modern styling, model comparison, and probability outputs.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox

# plotting
import matplotlib.pyplot as plt

from utils import (
    load_model, load_scaler, load_all_models,
    FEATURE_ORDER, prepare_sample, evaluate_accuracies
)

# ---------- Main Application ----------
class CHDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Load pre-trained models (fast startup)
        print("Loading pre-trained models...")
        self.best_model = load_model()
        self.scaler = load_scaler()
        self.all_models = load_all_models()
        self.feature_order = FEATURE_ORDER
        
        # Evaluate accuracies on test data
        self.accuracies = evaluate_accuracies(self.all_models)
        self.best_model_name = max(self.accuracies.items(), key=lambda x: x[1])[0]
        print(f"Models loaded! Best model: {self.best_model_name}")
        
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
                               values=list(self.all_models.keys()), state="readonly")
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
            inputs = {}
            for col in self.feature_order:
                if col in self.inputs:
                    inputs[col] = self.inputs[col].get()
                else:
                    inputs[col] = "0"
            
            sample = prepare_sample(inputs, self.scaler)
            model = self.all_models[model_name]
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