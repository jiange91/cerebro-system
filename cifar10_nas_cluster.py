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

from pyspark import SparkConf

conf = SparkConf().setAppName('cluster') \
    .setMaster('spark://10.10.1.1:7077') \
    .set('spark.task.cpus', '16') \
    .set('spark.executor.memory', '124g')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.addPyFile("cerebro.zip")

# spark = SparkSession \
#     .builder \
#     .appName("Cerebro Example") \
#     .getOrCreate()

# ...
work_dir = '/var/nfs/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=6)
store = LocalStore(prefix_path=work_dir + 'test/')

from keras_tuner.engine import hyperparameters
import autokeras as ak
from cerebro.nas.hphpmodel import HyperHyperModel

img_shape = (32, 32, 3)

input_node = ak.ImageInput()
output_node = ak.ConvBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)
am = HyperHyperModel(input_node, output_node, seed=2000)
feature_columns = ['image']
label_columns = ['label']
am.resource_bind(
    backend=backend, 
    store=store,
    feature_columns=feature_columns,
    label_columns=label_columns,
    evaluation_metric='accuracy', 
)

am.tuner_bind(
    tuner="greedy", 
#     tuner="randomsearch",
    hyperparameters=None, 
    objective="val_accuracy",
    max_trials=20,
    overwrite=True,
    exploration=0.3,
)

with open ('/var/nfs/cifar10/prep_np/prep.npy', 'rb') as f:
    prep_x = np.load(f)
    prep_y = np.load(f)

rel = am.fit_on_prepared_data(prep_x=prep_x, prep_y=prep_y, batch_size=64, epochs=5, input_shape=img_shape)

import json
m = {}
for model in rel.metrics:
    m[model] = {}
    for key in rel.metrics[model]:
        if key != 'trial':
            m[model][key] = rel.metrics[model][key]
with open("mnist_nas_logs.txt", "w") as file:
    file.write(json.dumps(m))