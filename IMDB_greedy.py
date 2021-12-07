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

work_dir = '/var/nfs/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=6)
store = LocalStore(prefix_path=work_dir + 'test/')

feature_columns=['features']
label_columns=['label']

df = spark.read.load(work_dir + "IMDB/data.parquet")
df.show(5)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=100)

from keras_tuner.engine import hyperparameters
import autokeras as ak
from cerebro.nas.hphpmodel import HyperHyperModel

input_node = ak.TextInput()
output_node = ak.TextBlock(block_type="ngram")(input_node)
output_node = ak.ClassificationHead(num_classes=2, multi_label=True)(output_node)
am = HyperHyperModel(
    inputs=input_node, outputs=output_node, seed=2000
)

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
    max_trials=2,
    overwrite=True,
    exploration=0.3,
)

rel = am.fit(train_df, epochs=2)

import json
m = {}
for model in rel.metrics:
    m[model] = {}
    for key in rel.metrics[model]:
        if key != 'trial':
            m[model][key] = rel.metrics[model][key]
with open("IMDB_greedy_log.txt", "w") as file:
    file.write(json.dumps(m))