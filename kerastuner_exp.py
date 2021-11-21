import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras_tuner as kt


"""
The experiment for autokeras alone
Iris dataset
"""

data_path = "data/iris_training.csv"
dataframe = pd.read_csv(data_path, header=0)

print("Dataset shape: {}".format(dataframe.shape))

data = dataframe.values.astype("float32")
X, y = data[:, :-1], data[:, -1]

# Split into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def build_model(hp):
    inputs = tf.keras.Input(shape=4)
    x = inputs
    for layer in range(3):
        x = tf.keras.layers.Dense(
            units=hp.Int(f"units_{layer}", 64, 128, step=32), activation="relu",
        )(x)

    outputs = tf.keras.layers.Dense(units=3, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    max_trials=27*4,
    overwrite=True,
    objective="val_accuracy",
    directory="./keras_tuner_log/"
)

tuner.search(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=10
)
