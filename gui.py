"""
Basic GUI for Coronary Heart Disease Prediction
Loads pre-trained models from model/ directory.
"""

import warnings
warnings.filterwarnings("ignore")

from tkinter import *
import tkinter.messagebox as M

from utils import (
    load_model, load_scaler, load_all_models,
    FEATURE_ORDER, prepare_sample, evaluate_accuracies
)

# Load pre-trained models (fast startup)
print("Loading pre-trained models...")
best_model = load_model()
scaler = load_scaler()
all_models = load_all_models()
accuracies = evaluate_accuracies(all_models)
best_model_name = max(accuracies.items(), key=lambda x: x[1])[0]
print("Models loaded successfully!")

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
            
            sample = prepare_sample(inputs, scaler)
            result = best_model.predict(sample)[0]
            
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