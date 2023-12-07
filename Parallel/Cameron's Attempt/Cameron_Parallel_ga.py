# Cameron_Parallel_ga.py
# -*- coding: utf-8 -*-

import findspark
findspark.init()
import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from Cameron_Parallel_helperFunctions import (
    fitness_scoreRDD,
    selection_process,
    crossover,
    mutation,
    stopping_condition,
)

def main(args):
    # Load and preprocess data
    X, y = fetch_openml(args.dataset, return_X_y=True, as_frame=False)
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
    population_size = args.population_size
    num_features = X.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))

    # Parallelize the population as an RDD
    population_rdd = sc.parallelize(population)

    # Broadcast the model and evaluator to the cluster
    model_broadcast = sc.broadcast(lr)
    evaluator_broadcast = sc.broadcast(evaluator)

    # Main loop for the genetic algorithm
    for generation in range(args.evolution_rounds):
        print(f"Starting generation {generation}")

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
        print(f"Fitness scores for generation {generation}: {fitness_scores}")

        # Perform selection based on fitness scores
        selected_population = selection_process(fitness_scores, broadcast_population.value)
        new_population = crossover(selected_population)
        new_population = mutation(new_population)

        # Update the population for the next generation
        population = new_population

        # Check for convergence or other stopping criteria
        if stopping_condition(fitness_scores, args.stopping_threshold):
            print(f"Stopping condition met at generation {generation}")
            break

        # Destroy the broadcast variables to free resources
        broadcast_population.unpersist()

    # Destroy the broadcast variables to free resources and terminate Spark context
    model_broadcast.unpersist()
    evaluator_broadcast.unpersist()
    sc.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Genetic Algorithm')
    parser.add_argument('--dataset', default="gina_agnostic", help='Dataset Name')
    parser.add_argument('--population_size', type=int, default=100, help='Number of chromosomes')
    parser.add_argument('--evolution_rounds', type=int, default=50, help='Number of evolution rounds')
    parser.add_argument('--stopping_threshold', type=float, default=0.90, help='Stopping threshold for fitness score')
    args = parser.parse_args()
    
    # Initialize Spark context and session
    sc = SparkContext("local[*]", appName="Parallel Genetic Algorithm")
    spark = SparkSession(sc)
    main(args)