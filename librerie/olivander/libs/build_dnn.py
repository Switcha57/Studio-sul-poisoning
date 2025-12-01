import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Dense


def build_dnn(input_shape=2381):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(2381,)))
    model.add(Dense(512, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.losses.SparseCategoricalCrossentropy(name="loss"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return model


def fit_dnn(model, x, y, x_val, y_val):
    hist = model.fit(x, y, epochs=2000, validation_data=(x_val, y_val), batch_size=256)
    return model, hist

