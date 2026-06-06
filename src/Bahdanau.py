import tensorflow as tf
from tensorflow.keras.layers import Layer

class BahdanauAttention(Layer):

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)

        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units = units

    def get_config(self):
        # Cette méthode est cruciale pour sauvegarder/charger le modèle sans erreur
        config = super(BahdanauAttention, self).get_config()
        config.update({
            "units": self.units,
        })
        return config

    def call(self, values):

        # dernier hidden state
        hidden = values[:, -1, :]

        # (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score attention
        score = self.V(
            tf.nn.tanh(
                self.W1(values) +
                self.W2(hidden_with_time_axis)
            )
        )

        # attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # contexte
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights