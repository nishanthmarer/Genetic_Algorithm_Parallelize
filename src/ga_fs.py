def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from genetic_selection import fitness_score,fitness_population,select_metric,chromosome_selection,generate_next_population

import argparse
import numpy as np
import pandas as pd


from src.genetic_operations import mutation,crossover,crossover_population,generate_population

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer,fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def main(args):
    X, y = fetch_20newsgroups_vectorized(return_X_y=True)
    N,n_genes = X.shape
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

    metric = select_metric(args.metric_choice)

    population = generate_population(args.population_size,n_genes)

    for evo in np.arange(args.evolution_rounds):
        scores = fitness_population(X_tr,y_tr,X_te,y_te,
                           population,LogisticRegression,metric)

        print("Population {:3d} \t Score={:.3f}".format(evo,np.max(scores)))

        population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate)

        print(population)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm Sequential',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default=None,
                        help='Dataset Name')

    parser.add_argument('--crossover_choice', type=str,default='onepoint',
                        help='Crossover options for chromosomes (onepoint,multipoint)')

    parser.add_argument('--mutation_rate', type=float, default=0.5, help='Mutation rate for the chromosome')


    parser.add_argument('--metric_choice', type=str,default='accuracy',
                        help='Crossover options for chromosomes (f1,accuracy,roc_auc_score)')

    parser.add_argument('--population_size', type=int, default=500, help='Number of chromosomes to search over')

    parser.add_argument('--evolution_rounds', type=int, default=10, help='Number of evolution rounds to generate populations for')


    parser.add_argument('--stopping_threshold', type=float, default=0.99,
                        help='If the metric is above the stopping threshold, end search')



    args = parser.parse_args()

    main(args)