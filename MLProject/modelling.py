import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score, 
                             roc_curve, f1_score, precision_score, recall_score)

mlflow.set_experiment("CI Training Otomatis")
# Folder output sementara
os.makedirs('assets', exist_ok=True)

# Path Data
DATA_PATH = os.path.join('emails_preprocessing', 'emails_preprocessing.csv')

def run_experiment():
    print("🚀 Memulai Proses Training & Tuning...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} tidak ditemukan.")
        return

    df = pd.read_csv(DATA_PATH)

    # Buang kolom 'Email No.' jika masih ada, karena ini bikin ERROR string to float
    if 'Email No.' in df.columns:
        print("⚠️ Kolom 'Email No.' ditemukan. Sedang dihapus...")
        df = df.drop(columns=['Email No.'])
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['Prediction'])
    y = df['Prediction']

    print(f"✅ Data Siap. Ukuran: {X.shape}")

    # 2. Split Data
    # Stratify agar seimbang
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Pipeline & Parameter
    # Langsung masuk TfidfTransformer -> RandomForest
    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),                   # Ubah counts jadi bobot TF-IDF
        ('clf', RandomForestClassifier(random_state=42)) # Model
    ])

    # Skenario Tuning (Hyperparameter Tuning)
    param_grid = {
        'clf__n_estimators': [100],           # Epoch
        'clf__max_depth': [10, 20, 30],       # Cegah Overfitting
        'clf__min_samples_leaf': [1, 2, 4]    # Regularisasi
    }

    # 4. Eksekusi Tuning dengan MLflow Tracking
    with mlflow.start_run(run_name="Tuning_RandomForest_Fix"):
        print("⚙️ Sedang melakukan GridSearch (Tuning)...")
        
        # Grid Search
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        print(f"✅ Tuning Selesai. Best Params: {best_params}")

        # 5. Evaluasi
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Hitung Metrik
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        metrics = {
            "accuracy": acc,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": auc
        }
        
        print(f"📊 Hasil Akhir: Accuracy={acc:.4f}, AUC={auc:.4f}")

        # LOGGING KE DAGSHUB
        print("☁️ Uploading logs to DagsHub...")

        # Log Params & Metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model_rf_final")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.ylabel("Actual"); plt.xlabel("Predicted")
        cm_path = os.path.join("assets", "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.legend()
        roc_path = os.path.join("assets", "roc_curve.png")
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(roc_path)

        print("✅ Proses Selesai! Cek DagsHub Anda.")

if __name__ == "__main__":
    run_experiment()