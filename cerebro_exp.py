import os
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import LocalStore
from cerebro.tune import RandomSearch
from cerebro.tune import hp_choice, hp_quniform

os.environ["PYSPARK_PYTHON"] = '/usr/bin/python3.6'

"""
The convergence speed experiment of Cerebro vs AutoKeras
Iris dataset
"""

spark = SparkSession \
    .builder \
    .appName("Cerebro Example") \
    .getOrCreate()

# Load dataset
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df = spark.read.csv('./data/Iris_clean.csv', header=True, inferSchema=True)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
df = df.drop('SepalLengthCm').drop('SepalWidthCm').drop('PetalLengthCm').drop('PetalWidthCm')


# Resources
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path='/cerebro_logs/experiment/')

trials = []
node_numbers = [32, 64, 96, 128]
for num1 in node_numbers:
    for num2 in node_numbers:
        for num3 in node_numbers:
            trials.append((num1, num2, num3))


for layer1_num, layer2_num, layer3_num in trials:
    def estimator_gen_fn(params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=4, name='features'))
        model.add(tf.keras.layers.Dense(layer1_num, input_dim=(4,), activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=params["dropout_rate"]))
        model.add(tf.keras.layers.Dense(layer2_num, input_dim=layer1_num, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=params["dropout_rate"]))
        model.add(tf.keras.layers.Dense(layer3_num, input_dim=layer2_num, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=params["dropout_rate"]))
        model.add(tf.keras.layers.Dense(3, input_dim=layer3_num, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
        loss = 'sparse_categorical_crossentropy'

        estimator = SparkEstimator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            metrics=['acc'],
            batch_size=params['batch_size'])

        return estimator


    search_space = {
        'lr': hp_choice([1e-2, 1e-3, 13-4]),
        'batch_size': hp_quniform(16, 64, 16),
        'dropout_rate': hp_choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    }

    tuner = RandomSearch(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn,
                         search_space=search_space, num_models=72, num_epochs=20, validation=0.2,
                         evaluation_metric='accuracy', feature_columns=["features"], label_columns=['Species'])

    model = tuner.fit(df)
    model_history = model.get_history()
