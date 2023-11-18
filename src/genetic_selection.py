def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearnex import patch_sklearn
patch_sklearn()

from src.genetic_operations import mutation,crossover,crossover_population,generate_population
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
from tqdm import tqdm
def data_chromosome_subset(X,chromosome):
    """
    :param X: The dataset as an numpy array
    :param chromosome: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :return:
    """
    return X[:,chromosome.ravel()==1]

def fitness_score(X_tr,y_tr,X_te,y_te,chromosome,model,metric):
    """
    :param X_tr: The train dataset as an numpy array
    :param y_tr: The train label predictions as numpy array
    :param X_te: The train dataset as an numpy array
    :param y_te: The train label predictions as numpy array
    :param chromosome: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param model: scikit learn estimator
    :param model: scikit learn metric that accepts (y_true,y_predicted)
    :return:
    """
    # reshape chromosome incase it has wrong dimension...
    X_tr_subset = data_chromosome_subset(X_tr,chromosome)
    X_te_subset = data_chromosome_subset(X_te,chromosome)
    clf = model(random_state=123,n_jobs=-2)
    clf.fit(X_tr_subset,y_tr)
    y_pr = clf.predict(X_te_subset)
    return metric(y_te,y_pr)

def fitness_population(X_tr,y_tr,X_te,y_te,population,model,metric,verbose=False):
    """
    :param X_tr: The train dataset as an numpy array
    :param y_tr: The train label predictions as numpy array
    :param X_te: The train dataset as an numpy array
    :param y_te: The train label predictions as numpy array
    :param population: a # chromosomes x # genes numpy array of 0s and 1s , where a row is np.array([1.0,0.0,...,1.0])
    :param model: scikit learn estimator
    :param model: scikit learn metric that accepts (y_true,y_predicted)
    :return:
    """
    n_chromosomes,n_genes = population.shape
    scores = np.empty((n_chromosomes,),dtype=float)

    if verbose:
        slice = tqdm(range(n_chromosomes))
    else:
        slice = range(n_chromosomes)

    for n in slice:
        scores[n] = fitness_score(X_tr,y_tr,X_te,y_te,population[[n],:],model,metric)

    return scores

def chromosome_selection(scores,population):
    """
    :param scores: The scores of each chromosome in the population
    :param population: a # chromosomes x # genes numpy array of 0s and 1s , where a row is np.array([1.0,0.0,...,1.0])
    :return:
    """
    n_chromosomes,n_genes = population.shape
    top_scores_idx = np.argsort(scores)[::-1]

    return population[top_scores_idx[:n_chromosomes//2]]


def generate_next_population(scores,population,crossover_method="onepoint",mutation_rate=0.1,elitism=2):
    n_chromosomes,n_genes = population.shape
    # select fittest chromosomes in the population
    chromosome_fittest = chromosome_selection(scores,population)

    # cross over RANDOMLY!!!
    chromosome_cross = crossover_population(chromosome_fittest,n_chromosomes-elitism,crossover_method)

    # mutation
    chromosome_mutate = mutation(chromosome_cross,mutation_rate)

    return np.vstack((chromosome_fittest[:elitism,:],chromosome_mutate))


def select_metric(metric_choice):
    if metric_choice == "accuracy":
        return accuracy_score
    elif metric_choice == "f1":
        return f1_score
    elif metric_choice == "roc_auc_score":
        return roc_auc_score

if __name__ == "__main__":
    n_genes = 10
    n_chromosomes = 10000
    chromosome1 = np.random.randint(0,2,(1,n_genes))
    chromosome2 = np.random.randint(0,2,(1,n_genes))


    print('\n',''"="*10,"Check Dataset Selection Mutation","="*10)
    X = np.tile(np.arange(n_genes),(5,1))
    data_subset = data_chromosome_subset(X,chromosome1)
    print("Data: ",X)
    print("Chromosome: ",chromosome1)
    print("Data subset: ",data_subset)


    X, y = load_breast_cancer(return_X_y=True)
    N,n_genes = X.shape
    print('\n',''"="*10,"Check Fitness Score with Metric and Model","="*10)
    chromosome1 = np.random.randint(0,2,(1,n_genes))
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
    score =  fitness_score(X_tr,y_tr,X_te,y_te,chromosome1,LogisticRegression,f1_score)
    print("Logistic Regression Score with single chromosome: ",score)

    population = generate_population(1000,n_genes)
    scores = fitness_population(X_tr, y_tr, X_te, y_te, population, LogisticRegression, f1_score)
    print("Logistic Regression Score with Population: ",scores)