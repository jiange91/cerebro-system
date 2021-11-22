from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods.
from cerebro.tune import GridSearch, RandomSearch, TPESearch

# Utility functions for specifying the search space.
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

import tensorflow as tf
# tf.config.run_functions_eagerly(True)

from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"] = '/usr/bin/python3.6'

spark = SparkSession \
    .builder \
    .appName("Cerebro Iris") \
    .getOrCreate()

...

backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path='/cerebro_logs/experiment/')

from pyspark.ml.feature import OneHotEncoderEstimator

df = spark.read.csv("./data/Iris_clean.csv", header=True, inferSchema=True)

encoder = OneHotEncoderEstimator(dropLast=False)
encoder.setInputCols(["Species"])
encoder.setOutputCols(["Species_OHE"])

encoder_model = encoder.fit(df)
encoded = encoder_model.transform(df)

feature_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
label_columns=['Species_OHE']

# Initialize input DataFrames.
# You can download sample dataset from https://apache.googlesource.com/spark/+/master/data/mllib/sample_libsvm_data.txt

train_df, test_df = encoded.randomSplit([0.8, 0.2])

# Define estimator generating function.
# Input: Dictionary containing parameter values
# Output: SparkEstimator
def estimator_gen_fn(params):
    inputs = [tf.keras.Input(shape=(1,)) for col in feature_columns]
    concat = tf.keras.layers.Concatenate()(inputs)
    layer1_output = tf.keras.layers.Dense(32, activation="relu")(concat)
    layer2_output = tf.keras.layers.Dense(32, activation="relu")(layer1_output)
    output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(layer2_output)
    model = tf.keras.Model(inputs, output)

#     inputs = tf.keras.Input(shape=(4,))
#     output1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)
#     output2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(output1)
#     output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(output2)
#     model = tf.keras.Model(inputs, output)

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss = 'categorical_crossentropy'

    estimator = SparkEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        batch_size=params['batch_size'])

    return estimator

# Define dictionary containing the parameter search space.
search_space = {
    'lr': hp_choice([0.01, 0.001, 0.0001]),
    'batch_size': hp_quniform(16, 64, 16)
}

# Instantiate TPE (Tree of Parzan Estimators a.k.a., HyperOpt) model selection object.
model_selection = RandomSearch(
    backend=backend, 
    store=store, 
    estimator_gen_fn=estimator_gen_fn, 
    search_space=search_space,
    num_models=20, 
    num_epochs=5, 
    validation=0.2, 
    evaluation_metric='accuracy',
    feature_columns=feature_columns,
    label_columns=label_columns
)

# Perform model selection. Returns best model.
model = model_selection.fit(train_df)
print(model.metrics)

# Inspect best model training history.
model_history = model.get_history()
print(model_history)
