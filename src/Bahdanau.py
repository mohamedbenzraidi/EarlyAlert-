import tensorflow as tf
from tensorflow.keras.layers import Layer

class BahdanauAttention(Layer):
    def __init__(selfself, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        #values shape : (batch_size, time_steps, hidden_size)
        #score
        score=self.V(tf.nn.tanh(self.W1(values)))

        attention_weights = tf.nn.softmax(score,axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
