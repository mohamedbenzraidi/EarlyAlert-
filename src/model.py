import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    GaussianNoise,
)
from tensorflow.keras.models import Model

from src.Bahdanau import BahdanauAttention

_l2 = tf.keras.regularizers.l2(2e-4)
_reg_lstm = tf.keras.regularizers.l2(5e-6)


def buid_model(input_shape):
    inputs = Input(shape=input_shape)
    # Bruit léger en entraînement seulement — réduit le sur-apprentissage sur le train
    x = GaussianNoise(0.02)(inputs)

    # Capacité réduite (train AUC ~1 / val AUC ~0.45 → modèle trop riche)
    x = Bidirectional(
        LSTM(
            36,
            return_sequences=True,
            kernel_regularizer=_reg_lstm,
            recurrent_dropout=0.12,
        )
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    # x = Bidirectional(
    #     LSTM(
    #         20,
    #         return_sequences=True,
    #         kernel_regularizer=_reg_lstm,
    #         recurrent_dropout=0.12,
    #     )
    # )(x)
    # x = Dropout(0.55)(x)

    # attention = BahdanauAttention(24)
    # try:
    #     context_vector, attention_weights = attention(x, x)
    # except TypeError:
    #     context_vector, attention_weights = attention([x, x])

    x = Dense(40, activation="relu", kernel_regularizer=_l2)(x)
    x = Dropout(0.5)(x)

    # x = Dense(20, activation="relu", kernel_regularizer=_l2)(x)
    # x = Dropout(0.45)(x)

    # outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)

    return model