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

from pyspark import SparkConf

conf = SparkConf().setAppName('cluster') \
    .setMaster('spark://10.10.1.1:7077') \
    .set('spark.task.cpus', '80')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.addPyFile("cerebro.zip")

work_dir = '/var/nfs/'
backend = SparkBackend(spark_context=spark.sparkContext, num_workers=4)
store = LocalStore(prefix_path=work_dir + 'test/')

TRAIN_NUM = 100000
TEST_NUM = 10000

train_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load(work_dir+'data/parquet/train/*.parquet')
test_df = spark.read.format("parquet").option('header', 'true').option('inferSchema', 'true')\
    .load(work_dir+'data/parquet/valid/*.parquet')

train_row_nums = train_df.count()
test_row_nums = test_df.count()

train_data_ratio = TRAIN_NUM / train_row_nums
test_data_ratio = TEST_NUM / test_row_nums

print("Use {:%} of training data, with {} rows in the original data".format(train_data_ratio, train_row_nums))
print("Use {:%} of testing data, with {} rows in the original data".format(test_data_ratio, test_row_nums))

train_df.printSchema()
test_df.printSchema()

train_df = train_df.limit(TRAIN_NUM)
from pyspark.sql.functions import rand 
train_df = train_df.orderBy(rand())

test_df = test_df.limit(TEST_NUM)

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

model = am.fit(train_df, epochs=10)
metrics = model.metrics


import json
m = {}
for model in rel.metrics:
    m[model] = {}
    for key in rel.metrics[model]:
        if key != 'trial':
            m[model][key] = rel.metrics[model][key]

with open("criteo_nas_dev/metrics.txt", "w") as file:
    file.write(json.dumps(m))