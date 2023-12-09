import numpy as np
import pandas as pd
import itertools
from iteration_utilities import random_combination

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
def crossover(population1,population2,method="onepoint"):
    """
    :param population1: a population size x # genes numpy array of 0s and 1s. A row is np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param population2: a population size x # genes numpy array of 0s and 1s. A row is np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param method: onepoint swaps the first half of a chromosome with the first half of another chromosome. multipoint interleaves the genes of chromosome 1 with chromosome 2
    :return: a 1xN numpy array of 0s and 1s where the first half of array is the first half of chromosome 1, and the second half of aarray is the second half of chromosome 2
    """
    # get number of genes in chromosome
    n_chromosomes,n_genes = population1.shape

    if method=="onepoint":
        # crossover of chromosome 1 and chromosome 1
        chromosomes_crossed = np.column_stack((population1[:,:n_genes//2],population2[:,n_genes//2:]))
    elif method=="multipoint":
        chromosomes_crossed = np.empty((n_chromosomes,n_genes),dtype=population1.dtype)
        chromosomes_crossed[:,0::2] = population1[:,0::2]
        chromosomes_crossed[:,1::2] = population2[:,1::2]

    return chromosomes_crossed

def crossover_population(population,size=None,method="onepoint"):
    """
    :param population: a # of chromosomes x # genes numpy array of 0s and 1s. A row is np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param size: the size that the cross over population should be
    :param method: onepoint swaps the first half of a chromosome with the first half of another chromosome. multipoint interleaves the genes of chromosome 1 with chromosome 2
    :return:
    """
    n_chromosomes,n_genes = population.shape

    assert size is not None, "Cannot have no size for population!!!"
    # if the population is an even number, crossover is no problem (every other pair crossed)

    # TODO: find faster way of this perhaps...
    # crossover_enumerations = list(itertools.permutations(np.arange(n_chromosomes),2))
    # np.random.shuffle(crossover_enumerations)
    # pop1idx, pop2idx = zip(*crossover_enumerations[:(size)])
    pop1idx = np.random.randint(0,n_chromosomes,size=size)
    pop2idx = np.random.randint(0,n_chromosomes,size=size)

    #
    # if n_chromosomes % 2 == 0:
    #     population1 = population[::2]
    #     population2 = population[1::2]
    #     population_crossed = crossover(population1,population2,method=method)
    # else:
    #     population1 = population[:-1:2]
    #     population2 = population[1:-1:2]
    #     population_crossed = crossover(population1, population2, method=method)

    population1 = population[pop1idx,:]
    population2 = population[pop2idx,:]
    population_crossed = crossover(population1,population2,method=method)

    return population_crossed


def mutation(population,mutation_rate):
    """
    :param population: a population size x # genes numpy array of 0s and 1s. A row is np.array([1.0,0.0,...,1.0]) where 1 inidicates include feature from dataset
    :param mutation_rate: a probability indicating whether to flip the 1 to a 0, or a 0 to a 1
    :return:
    """
    # get number of genes in chromosome
    n_chromosomes,n_genes = population.shape

    # generate 1s with probability mutation rate
    mutation_mask = np.random.rand(n_chromosomes,n_genes) < mutation_rate

    # flip the bits of 1 to 0, and 0 to 1 with probability mutation rate. This corresponds to a xor gate.
    chromosome_mutated = np.logical_xor(population,mutation_mask).astype(int)

    return chromosome_mutated


if __name__ == "__main__":
    n_genes = 10
    n_chromosomes = 10001
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

    population1 = generate_population(5,n_genes)
    population2 = generate_population(5,n_genes)
    populationC = crossover(population1,population2)
    print("Population 1: ",population1)
    print("Population 2: ",population2)
    print("Population Crossover: ",populationC)
    print('\n',"="*10,"Check Population Init","="*10)

    assert chromosome_crossed.shape == chromosome1.shape, "Chromosome crossed shape should equal"
    assert np.all(populationC[:,:n_genes//2] == population1[:,:n_genes//2]),"First half of chromosome crossed should equal chromosome"
    assert np.all(populationC[:,n_genes//2:] == population2[:,n_genes//2:]),"First half of chromosome crossed should equal chromosome"


    population0 = generate_population(n_chromosomes,n_genes)

    print("Unique Chromosomes: ",len(np.unique(population0,axis=1)))
    print("Population Dimensions: ",population0.shape)

    populationC = crossover_population(population0, size=n_chromosomes,method="onepoint")
    assert populationC.shape[0] == population0.shape[0]//2, "Population crossed number of chromosomes should work with odd amount"

    print('\n',''"="*10,"Check Chromosome Mutation","="*10)
    print("Chromsome1: ",chromosome1)
    print("Chromosome Mutated with 0.1 rate: ",mutation(chromosome1,0.1))
    print("Chromosome Mutated with 0.9 rate: ",mutation(chromosome1,0.9))

    population_0_mutated = mutation(population0,0.1)
    print("Mutation on population with rate 0.1, check: ",1-np.mean(population_0_mutated==population0,axis=1).mean())
    population_0_mutated = mutation(population0,0.9)
    print("Mutation on population with rate 0.9, check: ",1-np.mean(population_0_mutated==population0,axis=1).mean())

