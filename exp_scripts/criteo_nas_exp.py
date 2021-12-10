import os
import autokeras as ak
from pyspark.sql import SparkSession
from cerebro.backend import SparkBackend
from cerebro.storage import LocalStore
from cerebro.nas.hphpmodel import HyperHyperModel
from keras_tuner import HyperParameters

"""
The experiment for generating a trial with random architecture and hyperparameters at the same time
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
store = LocalStore(prefix_path='/cerebro_autokeras_exp/')

TRAIN_NUM = 1000
TEST_NUM = 100

train_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/train/*.parquet')
test_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load('./data/parquet/valid/*.parquet')

train_row_nums = train_df.count()
test_row_nums = test_df.count()

train_data_ratio = TRAIN_NUM / train_row_nums
test_data_ratio = TEST_NUM / test_row_nums

print("Use {:%} of training data, with {} rows in the original data".format(train_data_ratio, train_row_nums))
print("Use {:%} of testing data, with {} rows in the original data".format(test_data_ratio, test_row_nums))

train_df.printSchema()
test_df.printSchema()

train_df = train_df.limit(TRAIN_NUM)

test_df = test_df.limit(TEST_NUM)

# Define the search space
input_node = ak.StructuredDataInput()
otuput_node = ak.DenseBlock()(input_node)
output_node = ak.ClassificationHead()(otuput_node)

am = HyperHyperModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=10
)

am.resource_bind(backend=backend, store=store, label_columns=["labels"], feature_columns=["features"],
                 evaluation_metric='accuracy')

hp = HyperParameters()
am.tuner_bind("randomsearch", hyperparameters=hp)

print(train_df.count())

model = am.fit(train_df, epochs=5)
metrics = model.metrics

for trial in metrics:
    info = metrics[trial]
    del info["trial"]
    metrics[trial] = info


with open("cerebro_nas_result/metrics.txt", "w") as file:
    file.writelines(metrics)
