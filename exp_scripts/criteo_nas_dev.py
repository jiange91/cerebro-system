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
    .set('spark.task.cpus', '16')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.addPyFile("cerebro.zip")

work_dir = '/var/nfs/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=4)
store = LocalStore(prefix_path=work_dir + 'test/')

TRAIN_NUM = 10000
TEST_NUM = 1000

train_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load(work_dir+'data/parquet/train/train_0.parquet')
test_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load(work_dir+'data/parquet/valid/valid_0.parquet')

train_row_nums = train_df.count()
test_row_nums = test_df.count()

train_data_ratio = TRAIN_NUM / train_row_nums
test_data_ratio = TEST_NUM / test_row_nums

print("Use {:%} of training data, with {} rows in the original data".format(train_data_ratio, train_row_nums))
print("Use {:%} of testing data, with {} rows in the original data".format(test_data_ratio, test_row_nums))

# train_df.printSchema()
# test_df.printSchema()

train_df = train_df.limit(TRAIN_NUM)
from pyspark.sql.functions import rand 
train_df = train_df.orderBy(rand())

test_df = test_df.limit(TEST_NUM)

from keras_tuner.engine import hyperparameters
import autokeras as ak
from cerebro.nas.hphpmodel import HyperHyperModel

# Define the search space
input_node = ak.StructuredDataInput()
otuput_node = ak.DenseBlock()(input_node)
output_node = ak.ClassificationHead()(otuput_node)

am = HyperHyperModel(input_node, output_node, seed=2500)

am.resource_bind(
    backend=backend, 
    store=store,
    feature_columns=["features"],
    label_columns=['labels'],
    evaluation_metric='accuracy', 
)

am.tuner_bind(
    tuner="greedy", 
    hyperparameters=None, 
    objective="val_accuracy",
    max_trials=20,
    overwrite=True,
    exploration=0.3,
)

print(train_df.count())

rel = am.fit(train_df, epochs=10)

import json
m = {}
for model in rel.metrics:
    m[model] = {}
    for key in rel.metrics[model]:
        if key != 'trial':
            m[model][key] = rel.metrics[model][key]

with open("criteo_nas_dev/metrics.txt", "w") as file:
    file.write(json.dumps(m))