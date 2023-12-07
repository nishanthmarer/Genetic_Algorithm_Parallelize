# Parallel_RDD-based.py

import findspark
findspark.init()

import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from Parallel_RDD_based_helperFunctions import (
    transform_data,
    fitness_scoreRDD,
    selection_process,
    crossover,
    mutation,
    stopping_condition,
)

def main(args):
    # Load data
    X, y = fetch_openml(args.dataset, return_X_y=True, as_frame=False)
    X = X.astype(float)
    y = y.astype(float)

    # Convert data to RDD of LabeledPoints
    data_rdd = sc.parallelize([
        LabeledPoint(label, features) for features, label in zip(X, y)
    ])

    # Split the data into training and test RDDs
    train_rdd, test_rdd = data_rdd.randomSplit([0.7, 0.3], seed=42)

    # Initialize the population
    population_size = args.population_size
    num_features = X.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))

    # Main loop for the genetic algorithm
    for generation in range(args.evolution_rounds):
        print(f"Starting generation {generation}")

        # Parallelize the fitness calculation over the population RDD
        population_rdd = sc.parallelize(population)
        fitness_scores_rdd = population_rdd.map(lambda chromosome: fitness_scoreRDD(train_rdd, test_rdd, chromosome))
        fitness_scores = fitness_scores_rdd.collect()

        # Perform selection, crossover, and mutation
        selected_population = selection_process(fitness_scores, population)
        new_population = crossover(selected_population)
        population = mutation(new_population)

        # Check for stopping condition
        if stopping_condition(fitness_scores, args.stopping_threshold):
            print(f"Stopping condition met at generation {generation}")
            break

    sc.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Genetic Algorithm with RDDs')
    parser.add_argument('--dataset', default="gina_agnostic", help='Dataset Name')
    parser.add_argument('--population_size', type=int, default=100, help='Number of chromosomes')
    parser.add_argument('--evolution_rounds', type=int, default=50, help='Number of evolution rounds')
    parser.add_argument('--stopping_threshold', type=float, default=0.90, help='Stopping threshold for fitness score')
    
    args = parser.parse_args()
    sc = SparkContext(appName="Parallel Genetic Algorithm with RDDs")
    main(args)
