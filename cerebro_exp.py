import tensorflow as tf
from pyspark.sql import SparkSession
from cerebro.backend import SparkBackend
from cerebro.keras import SparkEstimator
from cerebro.storage import LocalStore
from cerebro.tune import RandomSearch
from cerebro.tune import hp_choice, hp_quniform


"""
The convergence speed experiment of Cerebro vs AutoKeras
Iris dataset
"""

spark = SparkSession \
    .builder \
    .appName("Cerebro Example") \
    .getOrCreate()

# Load dataset
df = spark.read.csv('./data/iris_training.csv', header=True)
train_df, test_df = df.randomSplit([0.8, 0.2])

# Resources
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=1)
store = LocalStore(prefix_path='./cerebro_logs/experiment/')

trials = []
node_numbers = [64, 96, 128]
for num1 in node_numbers:
    for num2 in node_numbers:
        for num3 in node_numbers:
            trials.append((num1, num2, num3))


for layer1_num, layer2_num, layer3_num in trials:
    def estimator_gen_fn(params):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=4, name='features'))
        model.add(tf.keras.layers.Dense(layer1_num, input_dim=4, activation="relu"))
        model.add(tf.keras.layers.Dense(layer2_num, input_dim=layer1_num, activation="relu"))
        model.add(tf.keras.layers.Dense(layer3_num, input_dim=layer2_num, activation="relu"))
        model.add(tf.keras.layers.Dense(3, input_dim=layer3_num, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
        loss = 'binary_crossentropy'

        estimator = SparkEstimator(
            model=model,
            optimizer=optimizer,
            loss=loss,
            metrics=['acc'],
            batch_size=params['batch_size'])

        return estimator


    search_space = {
        'lr': hp_choice([0.01, 0.001, 0.0001]),
        'batch_size': hp_quniform(16, 256, 16)
    }

    tuner = RandomSearch(backend=backend, store=store, estimator_gen_fn=estimator_gen_fn,
                         search_space=search_space, num_models=4, num_epochs=10, validation=0.15,
                         evaluation_metric='accuracy', feature_columns=['sepal width', 'sepal length', 'pedal width',
                                                                        'pedal length'],
                         label_columns=['labels'])

    model = tuner.fit(train_df)
    model_history = model.get_history()
