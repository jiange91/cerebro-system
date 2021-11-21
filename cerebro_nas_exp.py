from tensorflow.keras.datasets import mnist
import autokeras as ak
from pyspark.sql import SparkSession
from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore
from cerebro.nas.hphpmodel import HyperHyperModel
from keras_tuner import HyperParameters

"""
The experiment for autokeras + Cerebro
"""

# Build the SparkSession
spark = SparkSession.builder \
    .master("local") \
    .appName("Example") \
    .config("spark.executor.memory", "1gb") \
    .getOrCreate()

sc = spark.sparkContext

backend = SparkBackend(spark_context=sc, num_workers=1)
store = LocalStore(prefix_path='Your data directory')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Define the search space
input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    block_type="resnet",
    normalize=True,
    augment=False
)(input_node)
output_node = ak.ClassificationHead()(output_node)

am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=200
)

am.resource_bind(backend=backend, store=store)

hp = HyperParameters()
am.tuner_bind("gridsearch", hyperparameters=hp)

am.fit()

am.tuner.search_space_summary()

