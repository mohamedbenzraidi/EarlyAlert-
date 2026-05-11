import tensorflow as tf
from tensorflow.keras.layers import Layer


class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        # On ajoute **kwargs pour accepter les arguments de Keras (trainable, name, etc.)
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values=None):
        if values is None:
            values = query

        # État « décodeur » : dernier pas de temps (classif de séquence)
        query_state = query[:, -1:, :]
        # Bahdanau: score_i = V^T tanh(W1 h_i + W2 s)
        w1_h = self.W1(values)
        w2_s = self.W2(query_state)
        score = self.V(tf.nn.tanh(w1_h + w2_s))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        # Indispensable pour load_model
        config = super(BahdanauAttention, self).get_config()
        config.update({
            "units": self.units,
        })
        return config