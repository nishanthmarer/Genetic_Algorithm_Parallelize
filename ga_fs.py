def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from joblib import Parallel,delayed

# from sklearnex import patch_sklearn
# patch_sklearn()

from src.genetic_selection import fitness_population,select_metric,generate_next_population,fitness_score,chromosome_selection,data_chromosome_subset
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
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # create the filename and folder structure
    os.makedirs(os.path.join("results"),exist_ok=True)
    results_path = os.path.join("results")

    now = datetime.now()
    experiment_time = now.strftime("%d.%m.%Y_%H.%M.%S.%f")
    os.makedirs(os.path.join(os.path.join(results_path,experiment_time)),exist_ok=True)

    config_df = pd.DataFrame({k:[v] for k,v in args.__dict__.items()}).to_csv(os.path.join(results_path, experiment_time,"config.csv"),index=False)

    filename = os.path.join(results_path, experiment_time, "benchmark.csv")
    test_filename = os.path.join(results_path,experiment_time,"test_metrics.csv")


    print("= "*10,f"{args.dataset} Dataset Description"," ="*10)
    X_tr,X_val,X_te,y_tr,y_val,y_te = load_dataset(args.dataset)
    N_size,n_genes = X_tr.shape

    print("Dataset Train Shape: ",X_tr.shape)
    print("Dataset Val Shape: ",X_tr.shape)
    print("Dataset Test Shape: ",X_te.shape)
    print()

    metric = select_metric(args.metric_choice)

    population = generate_population(args.population_size,n_genes)
    start_time = time()
    model = select_model(args.model)
    clf = model(random_state=123)

    clf.fit(X_tr,y_tr)
    y_pr_te = clf.predict(X_te)
    y_pr_val = clf.predict(X_val)

    baseline_metric_val = metric(y_val,y_pr_val)
    baseline_metric_test = metric(y_te,y_pr_te)
    end_time = time()
    total_time = end_time - start_time

    print("= "*10,"All Features Baseline Fit"," ="*10)
    print("Baseline Fit Scores: Val {:.4f} \t Test {:.4f}".format(baseline_metric_val,baseline_metric_test))
    print("Time to complete={:.3f}".format(total_time))
    print()

    print("= "*10,args.algorithm," ="*10)

    if args.algorithm == "baseline_metrics":
        best_chromosome = np.ones(n_genes).astype(int)
        evo = 0

        csv_writer_util(filename, evo, total_time, baseline_metric_val, best_chromosome)

    elif args.algorithm == "rfs":
        start_time = time()
        clf = model(random_state=123)
        sfs = SequentialFeatureSelector(clf,n_jobs=-2)
        sfs.fit(X_tr,y_tr)
        end_time = time()
        total_time = end_time - start_time

        best_chromosome = sfs.support_.astype(int)

        sfs_metric = fitness_score(X_tr, y_tr, X_val, y_val, best_chromosome, model, metric)

        print("Forward-Backward RFS \t Score={:.3f} \t time={:.3f}".format(sfs_metric,total_time))
        print()

        evo = 0

        csv_writer_util(filename, evo, total_time, sfs_metric, best_chromosome)

    elif args.algorithm == "ga_seq":
        print("Genetic Algorithm Evolution")
        scores = baseline_metric_val
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:
            seed += 1
            start_time = time()
            scores = fitness_population(X_tr,y_tr,X_val,y_val,population,model,metric,verbose=False)

            best_chromosome = population[np.argmax(scores)]

            np.random.seed(seed); random.seed(seed)
            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            csv_writer_util(filename, evo, total_time, scores, best_chromosome)

            evo += 1

    elif args.algorithm == "ga_joblib":
        print("Genetic Algorithm Evolution with joblib")
        scores = baseline_metric_val
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:
            seed += 1
            start_time = time()

            n_chromosomes, n_genes = population.shape

            # prefer='threads'
            scores = Parallel(n_jobs=-2,prefer=args.backend_prefer,max_nbytes=100)(
                delayed(fitness_score)(X_tr, y_tr, X_val, y_val, population[[n], :], model, metric) for n in range(n_chromosomes))

            scores = np.array(scores)

            best_chromosome = population[np.argmax(scores)]
            np.random.seed(seed); random.seed(seed)
            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            csv_writer_util(filename, evo, total_time, scores, best_chromosome)


            evo += 1

    elif args.algorithm == "random":
        print("Random Feature Evolution")
        scores = baseline_metric_val
        evo = 0
        best_score = 0; best_chromosome = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:
            seed += 1
            start_time = time()
            population = generate_population(args.population_size, n_genes)
            scores = fitness_population(X_tr,y_tr,X_val,y_val,
                               population,model,metric,verbose=False)

            best_idx = np.argmax(scores)
            population_best_chromosome = population[np.argmax(scores)]

            if scores[best_idx] > best_score:
                best_chromosome = population[np.argmax(scores)]
                best_score = scores[best_idx]

            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))

            csv_writer_util(filename, evo, total_time, scores, population_best_chromosome)

            evo += 1

    elif args.algorithm == "ga_spark":
        import findspark
        findspark.init()
        from pyspark import SparkContext

        N = 50

        # Initialize Spark context
        sc = SparkContext(appName="Parallel Genetic Algorithm with Scikit-Learn")
        sc.setLogLevel("ERROR")
        #Since data is large we will broadcast the data to all nodes (one time)
        broadCast_X_tr = sc.broadcast(X_tr)
        broadCast_X_val = sc.broadcast(X_val)
        broadCast_y_tr = sc.broadcast(y_tr)
        broadCast_y_val = sc.broadcast(y_val)
        
        print("Start the spark process")
        print("Genetic Algorithm Evolution with spark backend")
        scores = baseline_metric_val
        evo = 0
        while np.max(scores) < args.stopping_threshold and evo <= args.evolution_rounds:
            seed += 1

            start_time = time()

            n_chromosomes, n_genes = population.shape
            
            # main spark code goes here
            # Parallelize the fitness calculation
            
            population_rdd = sc.parallelize(population,N)
            scores = population_rdd.map(lambda chromosome: fitness_score(broadCast_X_tr.value, broadCast_y_tr.value, broadCast_X_val.value, broadCast_y_val.value, 
                                                                                     chromosome, model, metric)).collect()
            scores = np.array(scores)
            
            best_chromosome = population[np.argmax(scores)]
            np.random.seed(seed); random.seed(seed)
            population = generate_next_population(scores,population,crossover_method=args.crossover_choice,mutation_rate=args.mutation_rate,elitism=args.elitism)
            end_time = time()
            total_time = end_time-start_time

            print("Generation {:3d} \t Population Size={} \t Score={:.3f} \t time={:2f}s".format(evo,population.shape,np.max(scores),total_time))
            
            csv_writer_util(filename, evo, total_time, scores, best_chromosome)
            evo += 1
            
        #Memory Management 
        broadCast_X_tr.unpersist()
        broadCast_X_val.unpersist()
        broadCast_y_tr.unpersist()
        broadCast_y_val.unpersist()
        
        # End spark context
        sc.stop()
    
    else:
        raise Exception("Sorry, not a valid argument to choose")

    print()
    print("= "*10,"Best Chromosome Test Scores"," ="*10)

    X_tr_subset = data_chromosome_subset(X_tr,best_chromosome)
    X_te_subset = data_chromosome_subset(X_te,best_chromosome)
    np.random.seed(123); random.seed(123)
    clf = model(random_state=123)
    clf.fit(X_tr_subset,y_tr)
    y_pr = clf.predict(X_te_subset)
    test_f1 =  select_metric("f1")(y_te,y_pr)
    test_accuracy=  select_metric("accuracy")(y_te,y_pr)
    test_roc_auc=  select_metric("roc_auc_score")(y_te,y_pr)
    print("Accuracy: {:.4f} \t F1: {:.4f} \t ROC-AUC: {:.4f}".format(test_accuracy,test_f1,test_roc_auc))
    print("Best Chromosome: ",best_chromosome)
    pd.DataFrame({"F1":[test_f1],"Accuracy":[test_accuracy],"ROC AUC":[test_roc_auc],"Best Chromosome":[best_chromosome]}).to_csv(test_filename,index=False)