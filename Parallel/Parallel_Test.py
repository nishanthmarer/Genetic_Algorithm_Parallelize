import pandas as pd
import numpy as np
import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from sklearn.datasets import fetch_openml
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors


sc = SparkContext("local[500]", appName='Parallel Sparse Logistic Regression')
spark = SparkSession(sc)
X, y = fetch_openml('gina_agnostic',return_X_y=True,as_frame=False)
X = X.astype(float); y=y.astype(float)

#data_pairs = list(zip(y.astype(float), X.astype(float)))
data_pairs = list(zip(X.astype(float), y.astype(float)))
rdd = sc.parallelize(data_pairs)
#rdd3 = rdd.map(lambda x:(((x[0]+1)/2),x[1]))
rdd3 = rdd.map(lambda x:(x[0],((x[1]+1)/2)))
#rdd2 = rdd3.map(lambda x:LabeledPoint(x[0],x[1]))
rdd2 = rdd2.repartition(100)

#rdd_rows = rdd3.map(lambda arr: Row(features=Vectors.dense(arr[0])))
#rdd_rows = rdd3.map(lambda row: Row(features=Vectors.dense(row[0]), label=float(row[1])))
rdd_rows = rdd3.map(lambda row: Row(features=Vectors.dense(list(zip([f"feature{i+1}" for i in range(X.shape[1])], row[0]))), label=float(row[1])))
df = spark.createDataFrame(rdd_rows, [f"feature{i+1}" for i in range(X.shape[1])] + ["target"])

vector_assembler = VectorAssembler(inputCols=[f"feature{i+1}" for i in range(X.shape[1])], outputCol="features")
sparkDF = vector_assembler.transform(rdd2).select("features", "target")

# Build the model
#model = LogisticRegressionWithLBFGS.train(rdd2)
model = LogisticRegressionModel.fit(rdd2)

# Evaluating the model on training data
labelsAndPreds = rdd2.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(rdd2.count())
print("Training Error = " + str(trainErr))
