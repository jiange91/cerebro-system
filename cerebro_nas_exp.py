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

backend = SparkBackend(spark_context=sc, num_workers=4)
store = LocalStore(prefix_path='cerebro_autokeras_exp')

train_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/train/*.parquet')
test_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/test/*.parquet')

train_df.printSchema()
test_df.printSchema()


# Define the search space
input_node = ak.StructuredDataInput()
output_node = ak.ClassificationHead()(input_node)

am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=8
)

am.resource_bind(backend=backend, store=store)

hp = HyperParameters()
am.tuner_bind("randomsearch", hyperparameters=hp)

am.fit(train_df, epochs=5)

am.tuner.search_space_summary()

