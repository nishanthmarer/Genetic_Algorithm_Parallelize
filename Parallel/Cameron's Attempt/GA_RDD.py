# GA_RDD.py
from sklearnex import patch_sklearn

patch_sklearn()
import findspark

findspark.init()
import csv
import datetime
import time
import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pyspark import SparkContext
from GA_RDD_helperFunctions import (
    select_metric,
    fitness_scoreRDD,
    selection_process,
    crossover,
    mutation,
    stopping_condition,
)


def main(args):
    total_start_time = time.time()

    metric_choice = args.metric_choice
    elitism = args.elitism
    crossover_choice = args.crossover_choice

    # Load and preprocess data
    X, y = fetch_openml(args.dataset, return_X_y=True, as_frame=False)
    X, y = X.astype(float), y.astype(float)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )

    # Train a Logistic Regression model on the full feature set
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-2, random_state=123)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute the Baseline Fit Score
    metric_function = select_metric(args.metric_choice)
    baseline_metric = metric_function(y_test, y_pred)
    print(f"Baseline Fit Score: {baseline_metric:.3f}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = (
        f"logs/GA_RDD_LogisticRegression_{args.metric_choice}_{timestamp}.csv"
    )
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(
            [
                "Baseline Fit Score",
                "Dataset",
                "Population Size",
                "Evolution Rounds",
                "Elitism",
                "Crossover Choice",
            ]
        )
        # Write initial information
        writer.writerow(
            [
                baseline_metric,
                args.dataset,
                args.population_size,
                args.evolution_rounds,
                args.elitism,
                args.crossover_choice,
            ]
        )

    # Initialize Spark context
    sc = SparkContext(appName="Parallel Genetic Algorithm with Scikit-Learn")

    # Broadcast the data
    broadcast_X = sc.broadcast(X)
    broadcast_y = sc.broadcast(y)

    # Initialize the population
    population_size = args.population_size
    num_features = X.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))

    # Main loop for the genetic algorithm
    for generation in range(args.evolution_rounds):
        start_time = time.time()
        print(f"Starting generation {generation}")

        if len(population) == 0:
            print("Population is empty, terminating the algorithm.")
            break

        print(f"Population size: {len(population)}")

        # Parallelize the fitness calculation
        population_rdd = sc.parallelize(population)
        fitness_scores_rdd = population_rdd.map(
            lambda chromosome: fitness_scoreRDD(
                broadcast_X.value, broadcast_y.value, chromosome, metric_choice
            )
        )
        fitness_scores = fitness_scores_rdd.collect()

        # Diagnostic information
        avg_fitness = np.mean(fitness_scores)
        best_fitness = np.max(fitness_scores)
        min_fitness = np.min(fitness_scores)
        print(
            f"Average fitness score: {avg_fitness}, Best fitness score: {best_fitness}, Worst fitness score: {min_fitness}"
        )

        best_chromosome = population[np.argmax(fitness_scores)]

        # Perform selection, crossover, and mutation
        selected_population = selection_process(fitness_scores, population, elitism)
        new_population = crossover(selected_population, crossover_choice)
        population = mutation(new_population)

        # Check for stopping condition
        if stopping_condition(fitness_scores, args.stopping_threshold):
            print(f"Stopping condition met at generation {generation}")
            break

        end_time = time.time()
        writer.writerow(
            [
                end_time - start_time,
                best_fitness,
                avg_fitness,
                min_fitness,
                "".join(map(str, best_chromosome)),
            ]
        )
        print(
            f"Time taken for generation {generation}: {end_time - start_time} seconds"
        )

    total_end_time = time.time()
    print(f"Total execution time: {(total_end_time - total_start_time)/60} minutes")
    # Stop Spark context
    sc.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel Genetic Algorithm with Scikit-Learn"
    )
    parser.add_argument("--dataset", default="gina_agnostic", help="Dataset Name")
    parser.add_argument(
        "--population_size", type=int, default=500, help="Number of chromosomes"
    )
    parser.add_argument(
        "--evolution_rounds", type=int, default=15, help="Number of evolution rounds"
    )
    parser.add_argument(
        "--stopping_threshold",
        type=float,
        default=0.99,
        help="Stopping threshold for fitness score",
    )
    parser.add_argument(
        "--metric_choice",
        type=str,
        default="accuracy",
        help="Metric choice for evaluating fitness (accuracy, f1, roc_auc_score)",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Number of fittest chromosomes to keep each population round",
    )
    parser.add_argument(
        "--crossover_choice",
        type=str,
        default="onepoint",
        help="Crossover method (onepoint, multipoint)",
    )

    args = parser.parse_args()
    main(args)
