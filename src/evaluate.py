from src.Bahdanau import BahdanauAttention
from src.losses import BinaryFocalLoss
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score
)
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import os

# from torchmetrics.functional import f1_score


def evaluate_model():
    #load model
    current_path = os.getcwd()  # Si tu es dans src/, root est le parent
    # Teste plutôt cette méthode plus robuste :
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "final_model.h5")
    X_test_path = os.path.join(BASE_DIR, "data", "processed", "X_test.npy")
    y_test_path = os.path.join(BASE_DIR, "data", "processed", "y_test.npy")
    # CHARGEMENT AVEC CUSTOM_OBJECTS
    model = load_model(
        model_path,
        custom_objects={
            "BahdanauAttention": BahdanauAttention,
            "BinaryFocalLoss": BinaryFocalLoss,
        },
        compile=False,
    )

    #load test data
    X_test = np.load(X_test_path)
    y_test = np.load(
        y_test_path
    )

    #evaluate
    y_score = model.predict(X_test).ravel()
    threshold = 0.5  # seuil de décision par défaut
    y_pred = (y_score >= threshold).astype(int)
    AUC = roc_auc_score(y_test, y_score)
    print(f"AUC: {AUC:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_score) #tpr = sensibilité,

    plt.plot(fpr, tpr, label=f"AUC = {AUC:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # ligne aléatoire
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    f1 = f1_score(y_test, y_pred)

    print("\n===== Clinical Metrics =====")

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()