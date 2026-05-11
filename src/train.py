import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.losses import BinaryFocalLoss
from src.model import buid_model

# Racine du projet (indépendante du répertoire de lancement)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA = os.path.join(_ROOT, "data", "processed")
_MODELS = os.path.join(_ROOT, "models")


def train_model():
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU TensorFlow:", gpus)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    if not gpus:
        print(
            "Note RTX 4060 : TensorFlow n'utilise pas CUDA sur Windows natif (TF 2.11+). "
            "Pour le GPU : WSL2 Ubuntu + pip install tensorflow[and-cuda], "
            "ou le plugin tensorflow-directml (versions compatibles à vérifier sur tensorflow.org)."
        )

    # Load data
    X_train = np.load(os.path.join(_DATA, "X_train.npy"))
    y_train = np.load(os.path.join(_DATA, "y_train.npy"))
    X_val = np.load(os.path.join(_DATA, "X_val.npy"))
    y_val = np.load(os.path.join(_DATA, "y_val.npy"))
    X_test = np.load(os.path.join(_DATA, "X_test.npy"))
    y_test = np.load(os.path.join(_DATA, "y_test.npy"))
    class_weight = np.load(os.path.join(_DATA, "class_weight.npy"), allow_pickle=True).item()
    class_weight = {int(float(k)): float(v) for k, v in class_weight.items()}
    for name, arr in ("X_train", X_train), ("X_val", X_val), ("X_test", X_test):
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(
                f"{name} contient NaN ou Inf. Régénérez les .npy : "
                "relancez prepare_data (étape 4 du pipeline) après mise à jour du preprocessing."
            )
    print(f"Type de X_train: {type(X_train)}")
    print(f"Type de y_train: {type(y_train)}")
    print(f"class_weight (fit): {class_weight}")

    pos_rate = float(np.mean(y_train))
    focal_alpha = float(np.clip(1.0 - pos_rate, 0.25, 0.75))
    # load model
    model = buid_model(X_train.shape[1:])
    try:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=5e-4,
            weight_decay=2e-4,
            clipnorm=1.0,
        )
    except AttributeError:
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)

    model.compile(
        optimizer=opt,
        loss=BinaryFocalLoss(gamma=2.0, alpha=focal_alpha),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )
    os.makedirs(_MODELS, exist_ok=True)
    # Arrêt dès que val_auc stagne (évite epochs où train → 1.0 et val s'effondre)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10,
        min_delta=1e-3,
        restore_best_weights=True,
    )
    # Réduire le LR quand val_auc plafonne (cohérent avec early stopping)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        min_delta=5e-4,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(_MODELS, "best_model.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    )

    print(f"Moyenne de X_train: {np.mean(X_train)}")
    print(f"Écart-type de X_train: {np.std(X_train)}")
    print(f"Min/Max de X_train: {np.min(X_train)} / {np.max(X_train)}")
    #train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
    )

    results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = float(results[0])

    p_val = model.predict(X_val, verbose=0).ravel()
    p_test = model.predict(X_test, verbose=0).ravel()
    y_va = y_val.astype(np.int32)
    y_te = y_test.astype(np.int32)

    test_auc = float(roc_auc_score(y_te, p_test))
    acc_default = float(accuracy_score(y_te, (p_test >= 0.5).astype(np.int32)))

    # Seuil F1 seul peut tomber à ~0.05 → prédictions quasi toujours positives (F1 artificiel, balanced acc ~0.5)
    thr_f1_only, best_f1_val = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 71):
        fv = f1_score(y_va, (p_val >= t).astype(np.int32), zero_division=0)
        if fv > best_f1_val:
            best_f1_val = fv
            thr_f1_only = float(t)

    thr_youden, best_j = 0.5, -1e9
    for t in np.linspace(0.15, 0.85, 71):
        pred = (p_val >= t).astype(np.int32)
        tp = int(np.sum((y_va == 1) & (pred == 1)))
        fn = int(np.sum((y_va == 1) & (pred == 0)))
        fp = int(np.sum((y_va == 0) & (pred == 1)))
        tn = int(np.sum((y_va == 0) & (pred == 0)))
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            thr_youden = float(t)

    # Seuil conservateur pour predict() : max (0.5*BA + 0.5*F1) sur la val, plage [0.15, 0.85]
    best_thr, best_combo = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 71):
        pred = (p_val >= t).astype(np.int32)
        ba = balanced_accuracy_score(y_va, pred)
        fv = f1_score(y_va, pred, zero_division=0)
        combo = 0.5 * ba + 0.5 * fv
        if combo > best_combo + 1e-6 or (
            abs(combo - best_combo) <= 1e-6 and t > best_thr
        ):
            best_combo = combo
            best_thr = float(t)

    pred_te = (p_test >= best_thr).astype(np.int32)
    test_acc_tuned = float(accuracy_score(y_te, pred_te))
    test_f1_tuned = float(f1_score(y_te, pred_te, zero_division=0))
    test_bal_acc = float(balanced_accuracy_score(y_te, pred_te))

    thr_path = os.path.join(_DATA, "inference_threshold.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "decision_threshold": best_thr,
                "threshold_youden_val": thr_youden,
                "threshold_f1_only_val": thr_f1_only,
                "tuned_on": "validation_balanced_f1_combo",
                "val_f1_at_saved_threshold": float(
                    f1_score(y_va, (p_val >= best_thr).astype(np.int32), zero_division=0)
                ),
                "val_balanced_accuracy_at_saved_threshold": float(
                    balanced_accuracy_score(y_va, (p_val >= best_thr).astype(np.int32))
                ),
            },
            f,
            indent=2,
        )

    print(f"Test Loss (Keras): {test_loss:.4f}")
    print(f"Test AUC (sklearn): {test_auc:.4f}")
    print(f"Test Accuracy @ seuil 0.5: {acc_default:.4f}")
    print(
        f"Test @ seuil τ={best_thr:.3f} (0.5×BA + 0.5×F1 sur val) : "
        f"acc={test_acc_tuned:.4f} | F1={test_f1_tuned:.4f} | balanced_acc={test_bal_acc:.4f}"
    )
    print(
        f"  (réf.) Youden val: τ={thr_youden:.3f} | F1-only (val, [0.15–0.85]): τ={thr_f1_only:.3f}, F1_val={best_f1_val:.3f}"
    )
    print(f"Seuil sauvegardé pour predict(): {thr_path}")

    metrics_path = os.path.join(_DATA, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_auc": test_auc,
                "test_accuracy_seuil_0.5": acc_default,
                "test_accuracy_seuil_optimal": test_acc_tuned,
                "test_f1_seuil_optimal": test_f1_tuned,
                "test_balanced_accuracy_seuil_optimal": test_bal_acc,
                "decision_threshold": best_thr,
            },
            f,
            indent=2,
        )
    print("Métriques test enregistrées:", metrics_path)

    final_path = os.path.join(_MODELS, "final_model.h5")
    print("Sauvegarde:", final_path)
    model.save(final_path)

