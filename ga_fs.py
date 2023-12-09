def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from joblib import Parallel,delayed

# from sklearnex import patch_sklearn
# patch_sklearn()

from src.genetic_selection import fitness_population,select_metric,generate_next_population,fitness_score,chromosome_selection
from src.genetic_operations import generate_population
from src.utils import load_dataset,select_model,csv_writer_util

import xgboost as xgb
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
                        help='Dataset Name (SantandereCustomerSatisfaction,IMDB.drama,gina_agnostic,hiva_agnostic,sylva_agnostic)')

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

    parser.add_argument('--model', type=str, default="logistic",
                        help='Type of model for fitting')

    parser.add_argument('--algorithm', type=str, default="ga_joblib",
                        help='Type of algorithm for feature selection (ga,rfs,random)')

    parser.add_argument('--backend_prefer',type=str,default="processes",help="backend preference for joblib  ('processes','threads') IGNORE -> (loky,threading)")



    args = parser.parse_args()

    # create the filename and folder structure
    os.makedirs(os.path.join("results"),exist_ok=True)
    results_path = os.path.join("results")

    now = datetime.now()
    experiment_time = now.strftime("%d.%m.%Y_%H.%M.%S.%f")
    os.makedirs(os.path.join(os.path.join(results_path,experiment_time)),exist_ok=True)

    config_df = pd.DataFrame({k:[v] for k,v in args.__dict__.items()}).to_csv(os.path.join(results_path, experiment_time,"config.csv"),index=False)

    filename = os.path.join(results_path, experiment_time, "benchmark.csv")


    # main(args)
    X_tr,X_te,y_tr,y_te = load_dataset(args.dataset)

    N,n_genes = X_tr.shape
    print("Dataset Train Shape: ",X_tr.shape)
    print("Dataset Test Shape: ",X_te.shape)

    metric = select_metric(args.metric_choice)

    population = generate_population(args.population_size,n_genes)
    start_time = time()
    model = select_model(args.model) # LogisticRegression(n_jobs=-2,random_state=123)
    clf = model(random_state=123) #xgb.XGBClassifier(random_state=123)
    # model = xgb

    clf.fit(X_tr,y_tr)
    y_pr = clf.predict(X_te)
    baseline_metric = metric(y_te,y_pr)
    end_time = time()

    print("Baseline Fit Score={:.3f}".format(baseline_metric))
    print("Time to complete={:.3f}".format(end_time-start_time))


    print("= "*10,args.algorithm," ="*10)

    if args.algorithm == "rfs":
        print()
        start_time = time()
        clf = model(random_state=123)
        sfs = SequentialFeatureSelector(clf,n_jobs=-2)
        sfs.fit(X_tr,y_tr)
        end_time = time()
        total_time = end_time - start_time
        X_tr_filtered = sfs.transform(X_tr)
        X_te_filtered = sfs.transform(X_te)

        clf.fit(X_tr_filtered,y_tr)
        y_pr = clf.predict(X_te_filtered)
        sfs_metric = metric(y_te,y_pr)

        best_chromosome = population[np.argmax(sfs_metric)]

        print("Forward-Backward RFS \t Score={:.3f} \t time={:.3f}".format(sfs_metric,total_time))
        print()

        evo = 0

        csv_writer_util(filename, evo, total_time, sfs_metric, best_chromosome)

    elif args.algorithm == "ga_seq":
        print("Genetic Algorithm Evolution")
        scores = baseline_metric
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:

            start_time = time()
            scores = fitness_population(X_tr,y_tr,X_te,y_te,population,model,metric,verbose=False)

            best_chromosome = population[np.argmax(scores)]

            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            csv_writer_util(filename, evo, total_time, scores, best_chromosome)


            
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
                delayed(fitness_score)(X_tr, y_tr, X_te, y_te, population[[n], :], model, metric) for n in range(n_chromosomes))

            scores = np.array(scores)

            best_chromosome = population[np.argmax(scores)]

            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            csv_writer_util(filename, evo, total_time, scores, best_chromosome)


            evo += 1

    elif args.algorithm == "random":
        print("Random Feature Evolution")
        scores = baseline_metric
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:

            start_time = time()
            population = generate_population(args.population_size, n_genes)
            scores = fitness_population(X_tr,y_tr,X_te,y_te,
                               population,model,metric,verbose=False)
            
            best_chromosome = population[np.argmax(scores)]

            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))

            csv_writer_util(filename, evo, total_time, scores, best_chromosome)

            evo += 1
    
    else:
        raise Exception("Sorry, not a valid argument to choose")