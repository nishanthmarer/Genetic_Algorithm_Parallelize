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
def crossover(chromosome1,chromosome2,method="onepoint"):
    """

    :param chromosome1: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param chromosome2: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :return: a 1xN numpy array of 0s and 1s where the first half of array is the first half of chromosome 1, and the second half of aarray is the second half of chromosome 2
    """
    # get number of genes in chromosome
    n_chromosomes,n_genes = chromosome1.shape

    if method=="onepoint":
        # crossover of chromosome 1 and chromosome 1
        chromosome_crossed = np.column_stack((chromosome1[:,:n_genes//2],chromosome2[:,n_genes//2:]))
    elif method=="multipoint":
        chromosome_crossed = np.empty((n_chromosomes,n_genes),dtype=chromosome1.dtype)
        chromosome_crossed[:,0::2] = chromosome1[:,0::2]
        chromosome_crossed[:,1::2] = chromosome2[:,1::2]

    return chromosome_crossed

def mutation(chromosome,mutation_rate):
    """
    :param chromosome: a 1xN numpy array of 0s and 1s np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param mutation_rate: a probability indicating whether to flip the 1 to a 0, or a 0 to a 1
    :return:
    """
    # get number of genes in chromosome
    n_chromosomes,n_genes = chromosome.shape

    # generate 1s with probability mutation rate
    mutation_mask = np.random.rand(n_chromosomes,n_genes) < mutation_rate

    # flip the bits of 1 to 0, and 0 to 1 with probability mutation rate. This corresponds to a xor gate.
    chromosome_mutated = np.logical_xor(chromosome,mutation_mask).astype(int)

    return chromosome_mutated



if __name__ == "__main__":
    n_genes = 10
    n_chromosomes = 10000
    chromosome1 = np.random.randint(0,2,(1,n_genes))
    chromosome2 = np.random.randint(0,2,(1,n_genes))

    print('\n',"="*10,"Check Chromosome Crossover","="*10)
    chromosome_crossed = crossover(chromosome1,chromosome2)

    assert chromosome_crossed.shape == chromosome1.shape, "Chromosome crossed shape should equal"
    assert np.all(chromosome_crossed[:,:n_genes//2] == chromosome1[:,:n_genes//2]),"First half of chromosome crossed should equal chromosome"
    assert np.all(chromosome_crossed[:,n_genes//2:] == chromosome2[:,n_genes//2:]),"First half of chromosome crossed should equal chromosome"

    chromosome_crossed = crossover(chromosome1,chromosome2,method="onepoint")
    print("Chromosome1: ",chromosome1)
    print("Chromosome2: ",chromosome2)
    print("ChromosomeC Onepoint: ",chromosome_crossed)

    chromosome_crossed = crossover(chromosome1,chromosome2,method="multipoint")
    print("Chromosome1: ",chromosome1)
    print("Chromosome2: ",chromosome2)
    print("ChromosomeC Multipoint: ",chromosome_crossed)

    print('\n',"="*10,"Check Population Init","="*10)
    population_0 = generate_population(n_chromosomes,n_genes)

    print("Unique Chromosomes: ",len(np.unique(population_0,axis=1)))
    print("Population Dimensions: ",population_0.shape)

    print('\n',''"="*10,"Check Chromosome Mutation","="*10)
    print("Chromsome1: ",chromosome1)
    print("Chromosome Mutated with 0.1 rate: ",mutation(chromosome1,0.1))
    print("Chromosome Mutated with 0.9 rate: ",mutation(chromosome1,0.9))

    population_0_mutated = mutation(population_0,0.1)
    print("Mutation on population with rate 0.1, check: ",1-np.mean(population_0_mutated==population_0,axis=1).mean())
    population_0_mutated = mutation(population_0,0.9)
    print("Mutation on population with rate 0.9, check: ",1-np.mean(population_0_mutated==population_0,axis=1).mean())