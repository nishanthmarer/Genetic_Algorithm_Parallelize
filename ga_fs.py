# ga_fs.py
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearnex import patch_sklearn
patch_sklearn()
from src.genetic_selection import fitness_population,select_metric,generate_next_population
from src.genetic_operations import generate_population
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer,fetch_20newsgroups_vectorized,fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from time import time
import csv
import datetime

random.seed(123)
np.random.seed(123)

def main(args):
    X, y = fetch_openml(args.dataset,return_X_y=True,as_frame=False)
    X = X.astype(float); y=y.astype(float)

    N,n_genes = X.shape
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

    metric = select_metric(args.metric_choice)

    population = generate_population(args.population_size,n_genes)

    clf = LogisticRegression(n_jobs=-2,random_state=123)
    clf.fit(X_tr,y_tr)
    y_pr = clf.predict(X_te)
    baseline_metric = metric(y_te,y_pr)

    print("Baseline Fit Score={:.3f}".format(baseline_metric))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"logs/{args.algorithm}_LogisticRegression_{args.metric_choice}_{timestamp}.csv"

    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write header information
        writer.writerow(['Baseline Fit Score', 'Dataset', 'Population Size', 'Evolution Rounds', 'Elitism', 'Crossover Choice', 'Mutation Rate'])
        writer.writerow([baseline_metric, args.dataset, args.population_size, args.evolution_rounds, args.elitism, args.crossover_choice, args.mutation_rate])
        writer.writerow(['Generation Time (s)', 'Best Fitness', 'Average Fitness', 'Worst Fitness', 'Best Chromosome'])

        if args.algorithm == "rfs":
            print()
            start_time = time()
            clf = LogisticRegression(n_jobs=-2,random_state=123)
            sfs = SequentialFeatureSelector(clf,n_jobs=-2)
            sfs.fit(X_tr,y_tr)
            end_time = time()
            total_time = end_time - start_time
            X_tr_filtered = sfs.transform(X_tr)
            X_te_filtered = sfs.transform(X_te)

            clf.fit(X_tr_filtered,y_tr)
            y_pr = clf.predict(X_te_filtered)
            sfs_metric = metric(y_te,y_pr)

            print("Forward-Backward RFS \t Score={:.3f} \t time={:.3f}".format(sfs_metric,total_time))
            print()
            total_time = end_time - start_time
            best_fitness = np.max(scores)
            avg_fitness = np.mean(scores)
            min_fitness = np.min(scores)
            best_chromosome = population[np.argmax(scores)]
            # Log generation details
            writer.writerow([evo, total_time, best_fitness, avg_fitness, min_fitness, ''.join(map(str, best_chromosome))])

        elif args.algorithm == "ga_sequential":
            print("Genetic Algorithm Evolution")
            for evo in np.arange(args.evolution_rounds):

                start_time = time()
                scores = fitness_population(X_tr,y_tr,X_te,y_te,
                                population,LogisticRegression,metric,verbose=False)

                population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
                end_time = time()
                total_time = end_time-start_time

                print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
                total_time = end_time - start_time
                best_fitness = np.max(scores)
                avg_fitness = np.mean(scores)
                min_fitness = np.min(scores)
                best_chromosome = population[np.argmax(scores)]
                # Log generation details
                writer.writerow([evo, total_time, best_fitness, avg_fitness, min_fitness, ''.join(map(str, best_chromosome))])

        elif args.algorithm == "random":
            print("Random Feature Evolution")
            for evo in np.arange(args.evolution_rounds):

                start_time = time()
                population = generate_population(args.population_size, n_genes)
                scores = fitness_population(X_tr,y_tr,X_te,y_te,
                                population,LogisticRegression,metric,verbose=False)

                end_time = time()
                total_time = end_time-start_time

                print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
                total_time = end_time - start_time
                best_fitness = np.max(scores)
                avg_fitness = np.mean(scores)
                min_fitness = np.min(scores)
                best_chromosome = population[np.argmax(scores)]
                # Log generation details
                writer.writerow([evo, total_time, best_fitness, avg_fitness, min_fitness, ''.join(map(str, best_chromosome))])
        else:
            raise Exception("Sorry, not a valid argument to choose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm Sequential',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default="gina_agnostic",
                        help='Dataset Name (Bioresponse,gina_agnostic,...)')

    parser.add_argument('--crossover_choice', type=str,default='onepoint',
                        help='Crossover options for chromosomes (onepoint,multipoint)')

    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate for the chromosome')


    parser.add_argument('--metric_choice', type=str,default='accuracy',
                        help='Crossover options for chromosomes (f1,accuracy,roc_auc_score)')

    parser.add_argument('--population_size', type=int, default=500, help='Number of chromosomes to search over')

    parser.add_argument('--elitism', type=int, default=2, help='Number fittest chromosomes to keep each population round')


    parser.add_argument('--evolution_rounds', type=int, default=15, help='Number of evolution rounds to generate populations for')


    parser.add_argument('--stopping_threshold', type=float, default=0.99,
                        help='If the metric is above the stopping threshold, end search')

    parser.add_argument('--algorithm', type=str, default="ga_sequential",
                        help='Type of algorithm for feature selection (ga,rfs,random)')



    args = parser.parse_args()

    main(args)