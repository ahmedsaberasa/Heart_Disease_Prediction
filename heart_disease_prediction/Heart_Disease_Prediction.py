# ===============================================================
# Heart Disease Prediction using Logistic Regression from Scratch
# Author: Ahmed Saber | GitHub: ahmedsaberasa
# Dataset: UCI Heart Disease (Cleveland) — 303 patients, 13 features
# ===============================================================

# ================= CELL 1 — Imports =================
# numpy         : numerical operations and matrix math
# pandas        : load and process CSV data
# matplotlib    : plotting (ROC curve, feature importance, confusion matrix)
# tkinter       : build the GUI window
# sklearn       : only for train_test_split and StandardScaler (no model!)
# pickle        : save/load trained model weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pickle
import os
import urllib.request


# ================= CELL 2 — Download & Load Data =================
DATA_URL  = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
DATA_FILE = "heart_disease.csv"
MODEL_CACHE = "heart_model_cache.pkl"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age":      "Age",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar >120 (1/0)",
    "restecg":  "Resting ECG Results (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1/0)",
    "oldpeak":  "ST Depression",
    "slope":    "Slope of Peak ST (0-2)",
    "ca":       "Number of Major Vessels (0-3)",
    "thal":     "Thal (0=Normal, 1=Fixed, 2=Reversible)"
}

def load_data():
    """Download dataset if not cached, then load."""
    if not os.path.exists(DATA_FILE):
        print("Downloading Heart Disease dataset...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Creating synthetic demo data (303 samples)...")
            _create_synthetic_data()
    
    df = pd.read_csv(DATA_FILE)
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Handle 'target' column — might be named 'condition' or 'num'
    for possible in ['target', 'condition', 'num', 'heart disease']:
        if possible in df.columns:
            df = df.rename(columns={possible: 'target'})
            break
    
    # Binarize target (UCI uses 0=no disease, 1-4=disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Keep only 13 features + target
    available = [f for f in FEATURE_NAMES if f in df.columns]
    df = df[available + ['target']].dropna()
    
    return df


def _create_synthetic_data():
    """Fallback: generate realistic synthetic data matching Cleveland stats."""
    np.random.seed(42)
    n = 303
    data = {
        'age':     np.random.normal(54, 9, n).clip(29, 77).astype(int),
        'sex':     np.random.binomial(1, 0.68, n),
        'cp':      np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08]),
        'trestbps':np.random.normal(131, 18, n).clip(94, 200).astype(int),
        'chol':    np.random.normal(246, 52, n).clip(126, 564).astype(int),
        'fbs':     np.random.binomial(1, 0.15, n),
        'restecg': np.random.choice([0,1,2], n, p=[0.48,0.02,0.50]),
        'thalach': np.random.normal(149, 23, n).clip(71, 202).astype(int),
        'exang':   np.random.binomial(1, 0.33, n),
        'oldpeak': np.abs(np.random.normal(1.0, 1.2, n)).round(1),
        'slope':   np.random.choice([0,1,2], n, p=[0.07,0.47,0.46]),
        'ca':      np.random.choice([0,1,2,3], n, p=[0.58,0.22,0.13,0.07]),
        'thal':    np.random.choice([0,1,2], n, p=[0.54,0.06,0.40]),
        'target':  np.random.binomial(1, 0.54, n)
    }
    pd.DataFrame(data).to_csv(DATA_FILE, index=False)


# ================= CELL 3 — Logistic Regression from Scratch =================

def sigmoid(z):
    """Sigmoid activation: maps any value to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y_true, y_pred):
    """Log loss: measures how far predictions are from true labels."""
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


def train_logistic_regression(X_train, y_train,
                               learning_rate=0.01,
                               epochs=2000,
                               lambda_reg=0.001):
    """
    Train Logistic Regression using Gradient Descent.
    
    Math:
      z     = X·w + b
      y_hat = sigmoid(z)
      Loss  = -mean(y·log(y_hat) + (1-y)·log(1-y_hat)) + (λ/2n)·||w||²
      ∂L/∂w = Xᵀ(y_hat - y) / n  +  λ·w / n
      ∂L/∂b = mean(y_hat - y)
    """
    n, m = X_train.shape
    w = np.zeros(m)       # weight for each feature
    b = 0.0               # bias term
    history = []

    for epoch in range(epochs):
        z     = X_train @ w + b
        y_hat = sigmoid(z)
        error = y_hat - y_train

        dw = (X_train.T @ error) / n + (lambda_reg * w) / n
        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 100 == 0:
            loss = binary_cross_entropy(y_train, y_hat)
            history.append(loss)

    return w, b, history


def predict(X, w, b, threshold=0.5):
    """Return class label (0 or 1) and probability."""
    prob = sigmoid(X @ w + b)
    return (prob >= threshold).astype(int), prob


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# ================= CELL 4 — Load Data & Train (or load cache) =================

def load_or_train():
    if os.path.exists(MODEL_CACHE):
        print("Loading cached model...")
        with open(MODEL_CACHE, 'rb') as f:
            return pickle.load(f)

    print("Loading and preparing data...")
    df = load_data()

    X = df[FEATURE_NAMES[:len([f for f in FEATURE_NAMES if f in df.columns])]].values
    y = df['target'].values

    # Scale features (important for gradient descent convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression from scratch...")
    w, b, loss_history = train_logistic_regression(
        X_train, y_train,
        learning_rate=0.05,
        epochs=3000,
        lambda_reg=0.01
    )

    y_pred_train, _ = predict(X_train, w, b)
    y_pred_test,  probs_test = predict(X_test, w, b)

    train_acc = accuracy(y_train, y_pred_train)
    test_acc  = accuracy(y_test,  y_pred_test)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Accuracy: {test_acc:.2f}%")

    # Compute ROC / AUC
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    roc_auc     = auc(fpr, tpr)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Feature importance (absolute weight values)
    feat_cols  = [f for f in FEATURE_NAMES if f in df.columns]
    importance = np.abs(w)

    result = dict(
        w=w, b=b, scaler=scaler,
        train_acc=train_acc, test_acc=test_acc,
        loss_history=loss_history,
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        cm=cm,
        feature_names=feat_cols,
        importance=importance,
        X_test=X_test, y_test=y_test, probs_test=probs_test
    )

    with open(MODEL_CACHE, 'wb') as f:
        pickle.dump(result, f)

    return result


# Run training
model = load_or_train()
print(f"\nModel ready!  Test Accuracy: {model['test_acc']:.2f}%  |  AUC: {model['roc_auc']:.3f}")


# ================= CELL 5 — GUI =================

BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
GREEN   = '#2ea043'
RED     = '#da3633'
BLUE    = '#388bfd'
MUTED   = '#8b949e'
WHITE   = '#e6edf3'
YELLOW  = '#d29922'

root = Tk()
root.title("Heart Disease Prediction — Ahmed Saber")
root.geometry("960x700")
root.configure(bg=BG)
root.resizable(False, False)

# ── Header ──
hdr = Frame(root, bg='#010409', pady=10)
hdr.pack(fill=X)
Label(hdr, text="🫀  Heart Disease Prediction",
      font=("Courier", 18, "bold"), bg='#010409', fg=WHITE).pack()
Label(hdr, text=f"Logistic Regression from Scratch  |  Test Acc: {model['test_acc']:.2f}%  |  AUC: {model['roc_auc']:.3f}",
      font=("Courier", 10), bg='#010409', fg=MUTED).pack()

# ── Main split ──
body = Frame(root, bg=BG)
body.pack(fill=BOTH, expand=True, padx=12, pady=8)

# LEFT — input form
left = Frame(body, bg=PANEL, relief='flat')
left.pack(side=LEFT, fill=Y, padx=(0, 8))

Label(left, text="Patient Data Input", font=("Courier", 11, "bold"),
      bg=PANEL, fg=BLUE).pack(pady=(10, 4))

form_frame = Frame(left, bg=PANEL)
form_frame.pack(padx=12)

entries = {}
defaults = {
    "age":52, "sex":1, "cp":0, "trestbps":125, "chol":212,
    "fbs":0, "restecg":1, "thalach":168, "exang":0,
    "oldpeak":1.0, "slope":2, "ca":2, "thal":3
}

feat_cols = model['feature_names']

for i, feat in enumerate(feat_cols):
    label_text = FEATURE_LABELS.get(feat, feat)
    Label(form_frame, text=label_text, font=("Courier", 9),
          bg=PANEL, fg=MUTED, anchor='w', width=30).grid(row=i, column=0, pady=2, sticky='w')
    var = StringVar(value=str(defaults.get(feat, 0)))
    e = Entry(form_frame, textvariable=var, font=("Courier", 10),
              bg='#0d1117', fg=WHITE, insertbackground=WHITE,
              relief='flat', width=8,
              highlightthickness=1, highlightbackground=BORDER,
              highlightcolor=BLUE)
    e.grid(row=i, column=1, pady=2, padx=(8, 0))
    entries[feat] = var

# ── Predict Button ──
result_var = StringVar(value="—")
prob_var   = StringVar(value="")

def run_predict():
    try:
        vals = []
        for feat in feat_cols:
            vals.append(float(entries[feat].get()))
        x = np.array(vals).reshape(1, -1)
        x_scaled = model['scaler'].transform(x)
        pred, prob = predict(x_scaled, model['w'], model['b'])
        p = float(prob[0]) * 100

        if pred[0] == 1:
            result_var.set(f"⚠  Heart Disease Detected")
            result_lbl.config(fg=RED)
        else:
            result_var.set(f"✔  No Heart Disease")
            result_lbl.config(fg=GREEN)
        prob_var.set(f"Probability: {p:.1f}%")
        prob_bar['value'] = p
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

btn = Button(left, text="⚡  Predict", command=run_predict,
             font=("Courier", 12, "bold"), bg=BLUE, fg='white',
             relief='flat', padx=20, pady=8, cursor='hand2',
             activebackground='#1f6feb')
btn.pack(pady=(12, 4))

result_lbl = Label(left, textvariable=result_var,
                   font=("Courier", 13, "bold"), bg=PANEL, fg=MUTED)
result_lbl.pack()
Label(left, textvariable=prob_var, font=("Courier", 10),
      bg=PANEL, fg=MUTED).pack()

prob_bar = ttk.Progressbar(left, length=260, maximum=100,
                            style='red.Horizontal.TProgressbar')
prob_bar.pack(pady=(4, 12))

style = ttk.Style()
style.theme_use('default')
style.configure('red.Horizontal.TProgressbar',
                troughcolor=BORDER, background=RED, thickness=10)

# RIGHT — charts
right = Frame(body, bg=BG)
right.pack(side=LEFT, fill=BOTH, expand=True)

fig = plt.Figure(figsize=(6.5, 6.2), facecolor=BG)
fig.subplots_adjust(hspace=0.45, wspace=0.4)
gs  = GridSpec(2, 2, figure=fig)

# Chart colors
ax_bg   = PANEL
txt_col = WHITE
grid_c  = BORDER

def style_ax(ax, title):
    ax.set_facecolor(ax_bg)
    ax.set_title(title, color=txt_col, fontsize=9, pad=6)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.title.set_fontfamily('monospace')

# 1 — Loss curve
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training Loss")
epochs_x = np.arange(len(model['loss_history'])) * 100
ax1.plot(epochs_x, model['loss_history'], color=BLUE, linewidth=1.5)
ax1.set_xlabel("Epoch", color=MUTED, fontsize=7)
ax1.set_ylabel("Log Loss", color=MUTED, fontsize=7)
ax1.grid(True, color=grid_c, linewidth=0.4)

# 2 — ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, f"ROC Curve (AUC = {model['roc_auc']:.3f})")
ax2.plot(model['fpr'], model['tpr'], color=GREEN, linewidth=1.8)
ax2.plot([0,1],[0,1], '--', color=MUTED, linewidth=1)
ax2.set_xlabel("False Positive Rate", color=MUTED, fontsize=7)
ax2.set_ylabel("True Positive Rate",  color=MUTED, fontsize=7)
ax2.grid(True, color=grid_c, linewidth=0.4)

# 3 — Confusion Matrix
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Confusion Matrix")
cm = model['cm']
im = ax3.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i,j]),
                 ha='center', va='center',
                 color=WHITE, fontsize=11, fontfamily='monospace')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Pred 0','Pred 1'], color=MUTED, fontsize=7)
ax3.set_yticklabels(['True 0','True 1'], color=MUTED, fontsize=7)

# 4 — Feature Importance
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Feature Importance (|weight|)")
imp   = model['importance']
feats = model['feature_names']
order = np.argsort(imp)
colors_bar = [GREEN if imp[i] == imp.max() else BLUE for i in order]
ax4.barh([feats[i] for i in order], imp[order],
          color=colors_bar, height=0.6)
ax4.set_xlabel("|weight|", color=MUTED, fontsize=7)
ax4.tick_params(axis='y', labelsize=6)
ax4.grid(True, axis='x', color=grid_c, linewidth=0.4)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.draw()
canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# ── Status bar ──
Label(root,
      text="💡  Enter patient data on the left and click Predict.  "
           "Charts show live model analytics.",
      font=("Courier", 8), bg='#010409', fg=MUTED
      ).pack(fill=X, side=BOTTOM, pady=3)

root.mainloop()
