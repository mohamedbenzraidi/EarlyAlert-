from src.model import build_model
# from src.preprocessing import load_data
import tensorflow as tf
import numpy as np
import os

root_path = os.getcwd()
os.makedirs(os.path.join(root_path, "models"), exist_ok=True)

def train_model():
    # Load data
    X_train = np.load(os.path.join(root_path, "data", "splits", "X_train.npy"))
    y_train = np.load(os.path.join(root_path, "data", "splits", "y_train.npy"))
    X_val = np.load(os.path.join(root_path, "data", "splits", "X_val.npy"))
    y_val = np.load(os.path.join(root_path, "data", "splits", "y_val.npy"))
    X_test = np.load(os.path.join(root_path, "data", "splits", "X_test.npy"))
    y_test = np.load(os.path.join(root_path, "data", "splits", "y_test.npy"))
    class_weight = np.load(os.path.join(root_path, "data", "splits", "class_weights.npy"), allow_pickle=True).item()
    # load model
    model = build_model(X_train.shape[1:])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    #callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(root_path, "models", "best_model.h5"),
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )

    #train
    history = model.fit(
        X_train, y_train,
        validation_data = (X_val,y_val),
        epochs=50,
        batch_size=64,
         class_weight=class_weight,

        callbacks=[early_stopping,reduce_lr, model_checkpoint]
        )

    #evaluate
    test_loss, test_acc, test_auc, test_recall = model.evaluate(X_test, y_test)

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_acc:.4f}, "
        f"Test AUC: {test_auc:.4f}, "
        f"Test Recall: {test_recall:.4f}"
    )
    # Sauvegarde du modèle



    model.save(
        os.path.join(root_path, "models", "final_model.h5")
    )


