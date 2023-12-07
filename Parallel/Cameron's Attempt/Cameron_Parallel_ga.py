# Cameron_Parallel_ga.py
# -*- coding: utf-8 -*-

import findspark
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
    findspark.init()

    sc = SparkContext("local[*]", appName="Parallel Genetic Algorithm")
    spark = SparkSession(sc)

    X, y = fetch_openml(args.dataset, return_X_y=True, as_frame=False)
    X = X.astype(float)
    y = y.astype(float)
    df_pandas = pd.DataFrame(
        data=np.column_stack([X, y]),
        columns=[f"feature{i+1}" for i in range(X.shape[1])] + ["target"],
    )
    sparkDF = spark.createDataFrame(df_pandas)
    sparkDF = sparkDF.withColumn("target", (col("target") + 1) / 2)

    vector_assembler = VectorAssembler(
        inputCols=[f"feature{i+1}" for i in range(X.shape[1])], outputCol="features"
    )
    sparkDF = vector_assembler.transform(sparkDF).select("features", "target")

    train_data, test_data = sparkDF.randomSplit([0.7, 0.3], seed=42)

    lr = LogisticRegression(featuresCol="features", labelCol="target")
    evaluator = BinaryClassificationEvaluator(
        labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    population_size = args.population_size
    num_features = X.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))

    population_rdd = sc.parallelize(population)

    model_broadcast = sc.broadcast(lr)
    evaluator_broadcast = sc.broadcast(evaluator)

    for generation in range(args.evolution_rounds):
        print(f"Starting generation {generation}")

        broadcast_population = sc.broadcast(population)

        fitness_scores_rdd = sc.parallelize(range(population_size)).map(
            lambda i: fitness_scoreRDD(
                train_data,
                test_data,
                broadcast_population.value[i],
                model_broadcast,
                evaluator_broadcast,
            )
        )

        fitness_scores = fitness_scores_rdd.collect()
        print(f"Fitness scores for generation {generation}: {fitness_scores}")

        selected_population = selection_process(fitness_scores, broadcast_population.value)
        new_population = crossover(selected_population)
        new_population = mutation(new_population)

        population = new_population

        if stopping_condition(fitness_scores, args.stopping_threshold):
            print(f"Stopping condition met at generation {generation}")
            break

        broadcast_population.unpersist()

    model_broadcast.unpersist()
    evaluator_broadcast.unpersist()
    sc.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Genetic Algorithm')
    parser.add_argument('--dataset', default="gina_agnostic", help='Dataset Name')
    parser.add_argument('--population_size', type=int, default=100, help='Number of chromosomes')
    parser.add_argument('--evolution_rounds', type=int, default=50, help='Number of evolution rounds')
    parser.add_argument('--stopping_threshold', type=float, default=0.95, help='Stopping threshold for fitness score')
    
    args = parser.parse_args()
    main(args)
