import numpy as np
import pandas as pd

def generate_population(n_chromosomes,n_genes):
    """
    :param n_chromosomes: the number of chromosomes to generate which create the population
    :param n_features: the number of genes in each chromosome (features) which creates the length of chromosome
    :return:
    """
    # the probability of any two chromosomes being equal is (.5)^(n_genes), which becomes quite small n_genes >> 10
    # the probability of a chromosome equal to any of N other chromosomes is (1-(1-(.5)^(n_genes))^N)
    population = np.random.randint(0,2,(n_chromosomes,n_genes))
    return population
def crossover(chromosome1,chromosome2):
    """

    :param chromosome1: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param chromosome2: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :return: a 1xN numpy array of 0s and 1s where the first half of array is the first half of chromosome 1, and the second half of aarray is the second half of chromosome 2
    """
    # get number of genes in chromosome
    n_chromosomes,n_genes = chromosome1.shape

    # crossover of chromosome 1 and chromosome 1
    chromosome_crossed = np.column_stack((chromosome1[:,:n_genes//2],chromosome2[:,n_genes//2:]))

    return chromosome_crossed




if __name__ == "__main__":
    n_genes = 10
    chromosome1 = np.random.randint(0,2,(1,n_genes))
    chromosome2 = np.random.randint(0,2,(1,n_genes))

    chromosome_crossed = crossover(chromosome1,chromosome2)

    assert chromosome_crossed.shape == chromosome1.shape, "Chromosome crossed shape should equal"
    assert np.all(chromosome_crossed[:,:n_genes//2] == chromosome1[:,:n_genes//2]),"First half of chromosome crossed should equal chromosome"
    assert np.all(chromosome_crossed[:,n_genes//2:] == chromosome2[:,n_genes//2:]),"First half of chromosome crossed should equal chromosome"

    print("Chromosome1: ",chromosome1)
    print("Chromosome2: ",chromosome2)
    print("ChromosomeC: ",chromosome_crossed)

    population_0 = generate_population(100,n_genes)

    print("Unique Chromosomes: ",len(np.unique(population_0,axis=1)))
    print("Population Dimensions: ",population_0.shape)