from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator

# datas storage for intermediate data and model artifacts.
from cerebro.storage import LocalStore, HDFSStore

# Model selection/AutoML methods.
from cerebro.tune import GridSearch, RandomSearch, TPESearch

# Utility functions for specifying the search space.
from cerebro.tune import hp_choice, hp_uniform, hp_quniform, hp_loguniform, hp_qloguniform
from cerebro.tune.base import ModelSelection, update_model_results

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark import SparkConf
from keras_tuner import HyperParameters
import autokeras as ak
from cerebro.nas.hphpmodel import HyperHyperModel
from keras_tuner.engine import hyperparameters
import os


os.environ["PYSPARK_PYTHON"] = '/usr/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/bin/python3.6'

conf = SparkConf().setAppName('training') \
    .setMaster('spark://training-cluster:7077') \
    .set('spark.task.cpus', '2')
spark = SparkSession.builder.config(conf=conf).getOrCreate()

...
work_dir = '/mnist_hp_exp/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=2)
store = LocalStore(prefix_path=work_dir + 'test/')

df = spark.read.format("libsvm") \
    .option("numFeatures", "784") \
    .load("data/mnist.scale") \



encoder = OneHotEncoderEstimator(dropLast=False)
encoder.setInputCols(["label"])
encoder.setOutputCols(["label_OHE"])

encoder_model = encoder.fit(df)
encoded = encoder_model.transform(df)

feature_columns = ['features']
label_columns = ['label_OHE']

train_df, test_df = encoded.randomSplit([0.8, 0.2], 100)

img_shape = (28, 28, 1)
num_classes = 10

hp = HyperParameters()
hp.Choice('optimizer', values=['adam'])
hp.Choice('learning_rate', values=[0.001, 0.0001])
hp.Choice('batch_size', values=[32, 64])

input_node = ak.ImageInput()
output_node = ak.ConvBlock(
    kernel_size=hyperparameters.Fixed('kernel_size', value=3),
    num_blocks=hyperparameters.Fixed('num_blocks', value=1),
    num_layers=hyperparameters.Fixed('num_layers', value=2),
)(input_node)
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
    hyperparameters=hp,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
)

_, _, meta_data, _ = am.sys_setup(train_df)

x = np.array(train_df.select(feature_columns).head(100))
y = np.array(train_df.select(label_columns).head(100))
x = [x[:, i] for i in range(x.shape[1])]
x = [r.reshape((-1, *img_shape)) for r in x]
y = np.squeeze(y, 1)

print(x[0].shape)
print(y.shape)

dataset, validation_data = am._convert_to_dataset(
    x=x, y=y, validation_data=None, batch_size=32
)

am._analyze_data(dataset)
am.tuner.hyper_pipeline = None
am.tuner.hypermodel.hyper_pipeline = None
tuner = am.tuner
tuner.hypermodel.hypermodel.set_fit_args(0.2, epochs=2)

hp = tuner.oracle.get_space()
tuner._prepare_model_IO(hp, dataset=dataset)
tuner.hypermodel.build(hp)
tuner.oracle.update_space(hp)
hp = tuner.oracle.get_space()

rel = tuner.fixed_arch_search(
    hp=hp,
    metadata=meta_data,
    epoch=2,
    x=dataset
)

