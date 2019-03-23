import numpy as np
from keras.layers import LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Lambda


def network_lstm(inputs, name):
    x = TimeDistributed(Bidirectional(LSTM(1024,
                                           return_sequences=True,
                                           kernel_initializer='he_normal',
                                           name=name)))(inputs)
    # x=Lambda(lambda x:K)
    # x = Bidirectional(LSTM(1024, return_sequences=True, kernel_initializer='he_normal'))(x)
    # x = Dense(2048, weights=[np.eye(2048)], use_bias=False, trainable=False)(x)
    return x
