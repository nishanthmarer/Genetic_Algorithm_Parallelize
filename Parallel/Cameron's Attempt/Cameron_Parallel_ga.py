# -*- coding: utf-8 -*-

import findspark

findspark.init()

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from Cameron_Parallel_helperFunctions import (
    fitness_scoreRDD,
    selection_process,
    crossover,
    mutation,
    stopping_condition,
)

# Initialize Spark context and session
sc = SparkContext("local[*]", appName="Parallel Genetic Algorithm")
spark = SparkSession(sc)

# Load and preprocess data
X, y = fetch_openml("gina_agnostic", return_X_y=True, as_frame=False)
X = X.astype(float)
y = y.astype(float)
df_pandas = pd.DataFrame(
    data=np.column_stack([X, y]),
    columns=[f"feature{i+1}" for i in range(X.shape[1])] + ["target"],
)
sparkDF = spark.createDataFrame(df_pandas)
sparkDF = sparkDF.withColumn("target", (col("target") + 1) / 2)

# Define a vector assembler to convert feature columns into a single vector column
vector_assembler = VectorAssembler(
    inputCols=[f"feature{i+1}" for i in range(X.shape[1])], outputCol="features"
)
sparkDF = vector_assembler.transform(sparkDF).select("features", "target")

# Split the data into training and test sets
train_data, test_data = sparkDF.randomSplit([0.7, 0.3], seed=42)

# Define the logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="target")

# Define the evaluator with the desired metric, e.g., areaUnderROC
evaluator = BinaryClassificationEvaluator(
    labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# Initialize the population (to be parallelized)
# For example purposes, let's say we have 10 features and want to generate a population of 100 chromosomes
population_size = 100
num_features = X.shape[1]
population = np.random.randint(2, size=(population_size, num_features))

# Parallelize the population as an RDD
population_rdd = sc.parallelize(population)

model_broadcast = sc.broadcast(lr)
evaluator_broadcast = sc.broadcast(evaluator)

# Main loop for the genetic algorithm
num_generations = 50  # Example number of generations
for generation in range(num_generations):
    # Broadcast the population to the cluster
    broadcast_population = sc.broadcast(population)

    # Map the fitness calculation over the population RDD using the index
    fitness_scores_rdd = sc.parallelize(range(population_size)).map(
        lambda i: fitness_scoreRDD(
            train_data,
            test_data,
            broadcast_population.value[i],
            model_broadcast,
            evaluator_broadcast,
        )
    )

    # Collect the fitness scores from the RDD
    fitness_scores = fitness_scores_rdd.collect()

    # Perform selection based on fitness scores
    selected_population = selection_process(fitness_scores, broadcast_population.value)

    # Perform crossover and mutation to generate a new population
    new_population = crossover(selected_population)
    new_population = mutation(new_population)

    # Update the population for the next generation
    population = new_population

    # Optionally, check for convergence or other stopping criteria
    if stopping_condition(fitness_scores):
        print(f"Stopping condition met at generation {generation}")
        break

    # Destroy the broadcast variables to free resources
    broadcast_population.unpersist()


# Close the broadcast variables
model_broadcast.unpersist()
evaluator_broadcast.unpersist()

# Terminate the Spark context
sc.stop()
