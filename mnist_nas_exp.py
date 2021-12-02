from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods.
from cerebro.tune import GridSearch, RandomSearch, TPESearch

# Utility functions for specifying the search space.
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyspark.sql import SparkSession
import numpy as np
import os

os.environ["PYSPARK_PYTHON"] = '/usr/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/bin/python3.6'


spark = SparkSession \
    .builder \
    .appName("Cerebro Example") \
    .getOrCreate()

...
work_dir = '/mnist_nas_exp/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path=work_dir + 'test/')

df = spark.read.format("libsvm") \
    .option("numFeatures", "784") \
    .load("data/mnist.scale")


from pyspark.ml.feature import OneHotEncoderEstimator

encoder = OneHotEncoderEstimator(dropLast=False)
encoder.setInputCols(["label"])
encoder.setOutputCols(["label_OHE"])

encoder_model = encoder.fit(df)
encoded = encoder_model.transform(df)

feature_columns = ['features']
label_columns = ['label_OHE']
train_df, test_df = encoded.randomSplit([0.8, 0.2], seed=100)

from keras_tuner.engine import hyperparameters
import autokeras as ak
from cerebro.nas.hphpmodel import HyperHyperModel

img_shape = (28, 28, 1)

input_node = ak.ImageInput()
output_node = ak.ConvBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)
am = HyperHyperModel(input_node, output_node, seed=2000)

am.resource_bind(
    backend=backend,
    store=store,
    feature_columns=feature_columns,
    label_columns=label_columns,
    evaluation_metric='accuracy',
)

am.tuner_bind(
    tuner="randomsearch",
    hyperparameters=None,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
)

rel = am.fit(train_df, epochs=2, input_shape=img_shape)


with open("mnist_nas_logs.txt", "w") as file:
    file.writelines(rel.metrics)

