from keras.layers import Dense, Dropout, BatchNormalization


def network_classification(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(81, activation='softmax', kernel_initializer='he_normal',name='cls_out')(x)
    return x

def network_regression(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(4, kernel_initializer='he_normal',name='reg_out')(x)
    return x
