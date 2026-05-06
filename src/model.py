from tensorflow.keras.layers import Input, Bidirectional, LSTM,Dense,Dropout, LayerNormalization
from tensorflow.keras.models import Model

from src.Bahdanau import BahdanauAttention


def buid_model(input_shape):
    inputs = Input(shape=input_shape)

    #BiLSTM1
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    #BiLSTM2
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    #Attention
    attention = BahdanauAttention(32)
    context_vector, attention_weights = attention(x)

    #dense layers
    x = Dense(64,activation="relu")(context_vector)
    x = Dropout(0.3)(x)

    x = Dense(32,activation="relu")(x)

    outputs = Dense(1, activation="sigmoid")(x)

    #model
    model = Model(inputs=inputs, outputs=outputs)

    return model