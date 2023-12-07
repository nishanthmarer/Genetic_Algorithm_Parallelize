# Parallel_RDD-based_helperFunctions.py

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def apply_chromosome(features, chromosome):
    return [feature if gene else 0.0 for feature, gene in zip(features, chromosome)]

def transform_data(rdd, chromosome):
    return rdd.map(lambda row: LabeledPoint(row.label, apply_chromosome(row.features, chromosome)))

def fitness_scoreRDD(train_rdd, test_rdd, chromosome):
    # Transform the data
    train_transformed = transform_data(train_rdd, chromosome)
    test_transformed = transform_data(test_rdd, chromosome)

    # Train the model
    model = LogisticRegressionWithLBFGS.train(train_transformed)

    # Predict and evaluate
    prediction_and_labels = test_transformed.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = BinaryClassificationMetrics(prediction_and_labels)
    fitness_score = metrics.areaUnderROC  # or any other suitable metric
    return fitness_score

def selection_process(fitness_scores, population):
    # Similar logic as before but adjusted for RDDs
    sorted_indices = np.argsort(fitness_scores)[::-1]
    return population[sorted_indices][:len(population) // 2]

def crossover(population, crossover_rate=0.7):
    # Perform crossover between pairs of chromosomes
    new_population = []
    for _ in range(len(population) // 2):
        parent1, parent2 = np.random.choice(len(population), 2, replace=False)
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(population[0]))
            child1 = np.concatenate(
                (
                    population[parent1][:crossover_point],
                    population[parent2][crossover_point:],
                )
            )
            child2 = np.concatenate(
                (
                    population[parent2][:crossover_point],
                    population[parent1][crossover_point:],
                )
            )
            new_population.extend([child1, child2])
        else:
            new_population.extend([population[parent1], population[parent2]])
    return np.array(new_population)


def mutation(population, mutation_rate=0.01):
    # Introduce random mutations into the population
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(len(population[i]))
            population[i][mutation_point] = 1 - population[i][mutation_point]
    return population


def stopping_condition(fitness_scores, threshold=0.95):
    # Check if any of the fitness scores meet the stopping threshold
    return np.any(fitness_scores >= threshold)
