def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from joblib import Parallel,delayed

from sklearnex import patch_sklearn
patch_sklearn()

from src.genetic_selection import fitness_population,select_metric,generate_next_population,fitness_score,chromosome_selection
from src.genetic_operations import generate_population
from src.utils import load_dataset

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
from time import time

random.seed(123)
np.random.seed(123)

# def main(args):



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genetic Algorithm Sequential',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default="SantandereCustomerSatisfaction",
                        help='Dataset Name (SantandereCustomerSatisfaction,IMDB.drama,...)')

    parser.add_argument('--crossover_choice', type=str,default='onepoint',
                        help='Crossover options for chromosomes (onepoint,multipoint)')

    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate for the chromosome')


    parser.add_argument('--metric_choice', type=str,default='accuracy',
                        help='Crossover options for chromosomes (f1,accuracy,roc_auc_score)')

    parser.add_argument('--population_size', type=int, default=200, help='Number of chromosomes to search over')

    parser.add_argument('--elitism', type=int, default=2, help='Number fittest chromosomes to keep each population round')


    parser.add_argument('--evolution_rounds', type=int, default=15, help='Number of evolution rounds to generate populations for')


    parser.add_argument('--stopping_threshold', type=float, default=0.99,
                        help='If the metric is above the stopping threshold, end search')

    parser.add_argument('--algorithm', type=str, default="ga_joblib",
                        help='Type of algorithm for feature selection (ga,rfs,random)')

    parser.add_argument('--backend_prefer',type=str,default="processes",help="backend for joblib (loky,threading)")



    args = parser.parse_args()

    os.makedirs('./results',exist_ok=True)
    results_path = os.path.join('./results')

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

    # main(args)
    X_tr,X_te,y_tr,y_te = load_dataset(args.dataset)

    N,n_genes = X_tr.shape
    print("Dataset Train Shape: ",X_tr.shape)
    print("Dataset Test Shape: ",X_te.shape)

    metric = select_metric(args.metric_choice)

    population = generate_population(args.population_size,n_genes)
    start_time = time()
    clf = LogisticRegression(n_jobs=-2,random_state=123)
    model = 'log'
    # clf = xgb.XGBClassifier(random_state=123)
    # model = xgb

    clf.fit(X_tr,y_tr)
    y_pr = clf.predict(X_te)
    baseline_metric = metric(y_te,y_pr)
    end_time = time()

    print("Baseline Fit Score={:.3f}".format(baseline_metric))
    print("Time to complete={:.3f}".format(end_time-start_time))

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

    elif args.algorithm == "ga":
        print("Genetic Algorithm Evolution")
        scores = baseline_metric
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:

            start_time = time()
            scores = fitness_population(X_tr,y_tr,X_te,y_te,population,LogisticRegression,metric,n_jobs=-2,verbose=False)
            # scores = fitness_population(X_tr,y_tr,X_te,y_te,population,xgb.XGBClassifier,metric,n_jobs=-2,verbose=False)

            best_chromosome = population[np.argmax(scores)]

            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            if os.path.isfile(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv'):
                data = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='a', header=False, index=False)
            else:
                columns = pd.DataFrame(columns = ['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'])
                result = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data = pd.concat((columns,result))
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='x', header=['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'], index=False)

            
            evo += 1

    elif args.algorithm == "ga_joblib":
        print("Genetic Algorithm Evolution with joblib")
        scores = baseline_metric
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:

            start_time = time()

            n_chromosomes, n_genes = population.shape

            # prefer='threads'
            scores = Parallel(n_jobs=-2,prefer=args.backend_prefer,max_nbytes=100)(
                delayed(fitness_score)(X_tr, y_tr, X_te, y_te, population[[n], :], LogisticRegression, metric, n_jobs=1) for n in range(n_chromosomes))

            # scores = Parallel(n_jobs=-2,prefer=args.backend_prefer,max_nbytes=100)(
            #     delayed(fitness_score)(X_tr, y_tr, X_te, y_te, population[[n], :], xgb.XGBClassifier, metric, n_jobs=1) for n in range(n_chromosomes))

            scores = np.array(scores)

            best_chromosome = population[np.argmax(scores)]

            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            if os.path.isfile(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv'):
                data = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='a', header=False, index=False)
            else:
                columns = pd.DataFrame(columns = ['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'])
                result = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data = pd.concat((columns,result))
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='x', header=['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'], index=False)

            evo += 1

    elif args.algorithm == "random":
        print("Random Feature Evolution")
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:

            start_time = time()
            population = generate_population(args.population_size, n_genes)
            scores = fitness_population(X_tr,y_tr,X_te,y_te,
                               population,LogisticRegression,metric,verbose=False)
            # scores = fitness_population(X_tr,y_tr,X_te,y_te,
            #                    population,xgb.XGBClassifier,metric,verbose=False)
            
            best_chromosome = population[np.argmax(scores)]

            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
    
            if os.path.isfile(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv'):
                data = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='a', header=False, index=False)
            else:
                columns = pd.DataFrame(columns = ['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'])
                result = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)], 'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
                data = pd.concat((columns,result))
                data.to_csv(f'{results_path}/{args.algorithm}_{model}_{args.dataset}_{args.metric_choice}_{dt_string}.csv', mode='x', header=['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'], index=False)

            evo += 1
    
    else:
        raise Exception("Sorry, not a valid argument to choose")