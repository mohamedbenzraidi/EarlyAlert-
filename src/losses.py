import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="EarlyAlert")
class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss pour classification binaire (Lin et al., 2017).
    Réduit le poids des exemples faciles et aide la généralisation (AUC) sur données bruitées.
    """

    def __init__(self, gamma=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        # Forcer (N, 1) pour éviter le broadcast (N,)×(N,1)→(N,N) et l'erreur Squeeze
        y_true = tf.cast(tf.reshape(y_true, [-1, 1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1, 1])
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        w = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        mod = tf.pow(1.0 - p_t, self.gamma)
        per = tf.reshape(mod * w * ce, [-1])
        return per

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg
