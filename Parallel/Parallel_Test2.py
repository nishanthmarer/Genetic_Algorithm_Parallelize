# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_openml

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

sc = SparkContext("local[60]", appName='Parallel Sparse Logistic Regression')

spark = SparkSession(sc)


X, y = fetch_openml('gina_agnostic',return_X_y=True,as_frame=False)
X = X.astype(float); y=y.astype(float)
#this will give a 2d array and 1d array
#combine these two to form a dataframe
df_pandas = pd.DataFrame(data=np.column_stack([X, y]), columns=[f"feature{i+1}" for i in range(X.shape[1])] + ["target"])

sparkDF = spark.createDataFrame(df_pandas)
sparkDF = sparkDF.withColumn("target", (col("target") + 1) / 2)
#sparkDF = sparkDF.repartition(100)
#sparkDF.count()

#ColumnLength = len(sparkDF.columns)
#RowsLength = sparkDF.count()

#Everytime change the vector_assembler and then pass the new data, it is fast 
#but maybe find a better way

vector_assembler = VectorAssembler(inputCols=[f"feature{i+1}" for i in range(X.shape[1])], outputCol="features")
sparkDF = vector_assembler.transform(sparkDF).select("features", "target")

train_data,test_data = sparkDF.randomSplit([0.7,0.3])

lr = LinearRegression(featuresCol = "features",labelCol="target")
trained_model = lr.fit(train_data)
unlabeled_data = test_data.select("features")

result = trained_model.evaluate(train_data)
predi = trained_model.transform(unlabeled_data)

print(result.r2)
#log = LogisticRegression(featuresCol = "features",labelCol="target")
#log_Trained = log.fit(train_data)
#train_results = log.evaluate(train_data)
#test_results = log.transform(test_data)

#print(test_results)
exit(0)
tp = test_results[(test_results.target==1)&(test_results.predicitions==1)].count()
tn = test_results[(test_results.target==0)&(test_results.predicitions==0)].count()

fp = test_results[(test_results.target==0)&(test_results.predicitions==1)].count()
fn = test_results[(test_results.target==1)&(test_results.predicitions==0)].count()

#acc = float((tp+tn)/(results.count()))

#print(acc)

