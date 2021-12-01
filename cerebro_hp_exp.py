import os
import autokeras as ak
from pyspark.sql import SparkSession
from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore
from cerebro.nas.hphpmodel import HyperHyperModel
import tensorflow as tf
import numpy as np


"""
The experiment for find the architecture first and then do hyperparameters tuning
"""

os.environ["PYSPARK_PYTHON"] = '/usr/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/bin/python3.6'

# Build the SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("Cerebro NAS Exp") \
    .config("spark.executor.memory", "1gb") \
    .getOrCreate()

sc = spark.sparkContext

backend = SparkBackend(spark_context=sc, num_workers=2)
store = LocalStore(prefix_path='/cerebro_hp_exp/')


train_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/train/*.parquet')
test_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/valid/*.parquet')

train_df.printSchema()
test_df.printSchema()

input_node = ak.ImageInput()
output_node = ak.DenseBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)

am = HyperHyperModel(input_node, output_node, overwrite=True, max_trials=8)

am.resource_bind(
    backend=backend,
    store=store,
    feature_columns=["features"],
    label_columns=["labels"],
    evaluation_metric='accuracy'
)

am.tuner_bind(
    tuner="randomsearch",
    hyperparameters=None,
    objective="val_accuracy",
    max_trials=20,
    overwrite=True,
)

x = np.array(train_df.select(["features"]).head(20000))
y = np.array(train_df.select(["labels"]).head(2000))

dataset, validation_data = am._convert_to_dataset(
    x=x, y=y, validation_data=None, batch_size=32
)

am._analyze_data(dataset)

tuner = am.tuner
tuner.hypermodel.hypermodel.set_fit_args(0.2, epochs=100)

hp = tuner.oracle.get_space()
tuner._prepare_model_IO(hp, dataset=dataset)
tuner.hypermodel.build(hp)
tuner.oracle.update_space(hp)
hp = tuner.oracle.get_space()

trial = tuner.oracle.create_trial(tuner.tuner_id)

tuner._prepare_model_IO(trial.hyperparameters, dataset=dataset)
model = tuner.hypermodel.build(trial.hyperparameters)
tuner.adapt(model, dataset)

model.summary()

optimizer = tf.keras.optimizers.Adam(lr=0.001)
params = {
    'model': model,
    'optimizer': optimizer, # keras opt not str
    'loss': 'categorical_crossentropy', # not sure
    'metrics': ['accuracy'],
    'batch_size': 64,
    'custom_objects': tf.keras.utils.get_custom_objects()
}

est = tuner.model_selection._estimator_gen_fn_wrapper(params)

ms = tuner.model_selection

_, _, metadata, _ = ms.backend.prepare_data(ms.store, train_df, ms.validation, label_columns=ms.label_cols, feature_columns=ms.feature_cols)
ms.backend.initialize_workers()
ms.backend.initialize_data_loaders(ms.store, None, ms.feature_cols + ms.label_cols)

train_rel = ms.backend.train_for_one_epoch([est], ms.store, None, ms.feature_cols, ms.label_cols)
