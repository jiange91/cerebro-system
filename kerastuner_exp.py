import tensorflow as tf
import pandas as pd
import keras_tuner as kt
from sklearn.preprocessing import OneHotEncoder


"""
The experiment for autokeras alone
Iris dataset
"""

data_path = "data/Iris_clean.csv"
dataframe = pd.read_csv(data_path, header=0)

print("Dataset shape: {}".format(dataframe.shape))

data = dataframe.values.astype("float32")
X, y = data[:, :-1], data[:, -1]
y = y.reshape(-1, 1)

encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()

def build_model(hp):
    inputs = tf.keras.Input(shape=4)
    x = inputs
    for layer in range(2):
        x = tf.keras.layers.Dense(
            units=hp.Int(f"units_{layer}", 32, 128, step=32), activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(
            rate=hp.Choice("dropout_rate", values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        )(x)

    outputs = tf.keras.layers.Dense(units=3, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        )
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    max_trials=16*72,  # To match with the cerebro experiment
    overwrite=True,
    objective="val_accuracy",
    directory="./keras_tuner_log/"
)

tuner.search(
    X,
    y,
    validation_split=0.20,
    epochs=20,
    callbacks=[tf.keras.callbacks.TensorBoard("./keras_tuner_log")]
)
