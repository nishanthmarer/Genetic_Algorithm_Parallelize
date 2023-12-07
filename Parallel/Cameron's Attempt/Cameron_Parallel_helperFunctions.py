from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import Broadcast
import numpy as np


# UDF for applying chromosome to the features
def apply_chromosome(features, chromosome):
    selected_features = [
        feature if gene else 0.0 for feature, gene in zip(features, chromosome)
    ]
    return Vectors.dense(selected_features)


udf_apply_chromosome = udf(apply_chromosome, VectorUDT())


def fitness_scoreRDD(train_data, test_data, chromosome, model, evaluator):
    # Define UDF inside the function to avoid serialization issues
    apply_chromosome_udf = udf(
        lambda features: apply_chromosome(features, chromosome), VectorUDT()
    )
    model = model_broadcast.value
    evaluator = evaluator_broadcast.value

    # Apply the chromosome to the features in the dataset
    train_data_transformed = train_data.withColumn(
        "features", apply_chromosome_udf(col("features"))
    )
    test_data_transformed = test_data.withColumn(
        "features", apply_chromosome_udf(col("features"))
    )

    # Fit the model and make predictions
    fitted_model = model.fit(train_data_transformed)
    predictions = fitted_model.transform(test_data_transformed)

    # Calculate and return the fitness score
    fitness_score = evaluator.evaluate(predictions)
    return fitness_score


def selection_process(fitness_scores, population):
    # Sort the population by fitness score and select the top individuals
    sorted_indices = np.argsort(fitness_scores)[::-1]
    selected_population = population[sorted_indices][: len(population) // 2]
    return selected_population


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
