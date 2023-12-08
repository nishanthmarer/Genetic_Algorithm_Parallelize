# GA_RDD_helperFunctions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


def select_metric(metric_choice):
    if metric_choice == "accuracy":
        return accuracy_score
    elif metric_choice == "f1":
        return f1_score
    elif metric_choice == "roc_auc_score":
        return roc_auc_score
    else:
        raise ValueError("Invalid metric choice")


def fitness_scoreRDD(X, y, chromosome, metric_choice):
    # Apply chromosome mask to select features
    selected_features = X[:, chromosome == 1]

    # Train the model
    # model = LogisticRegression(max_iter=1000, n_jobs=-2,random_state=123)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-2, random_state=123)
    )
    model.fit(selected_features, y)

    metric = select_metric(metric_choice)

    # Predict and evaluate
    predictions = model.predict(selected_features)
    fitness_score = metric(y, predictions)
    return fitness_score


def selection_process(fitness_scores, population, elitism):
    # Sort the population by fitness score
    sorted_indices = np.argsort(fitness_scores)[::-1]

    # Select top elitism chromosomes directly
    elite_population = population[sorted_indices[:elitism]]

    # Fill the rest of the selected population based on fitness scores
    remaining_population = population[sorted_indices][elitism : len(population) // 2]

    # Combine elite and remaining selected population
    selected_population = np.vstack((elite_population, remaining_population))

    return selected_population


def crossover(population, crossover_choice, crossover_rate=0.7):
    # Perform crossover between pairs of chromosomes based on the chosen method
    new_population = []

    for _ in range(len(population) // 2):
        parent1, parent2 = np.random.choice(len(population), 2, replace=False)

        if np.random.rand() < crossover_rate:
            if crossover_choice == "onepoint":
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
            elif crossover_choice == "multipoint":
                mask = np.random.randint(0, 2, size=len(population[0]), dtype=bool)
                child1 = np.where(mask, population[parent1], population[parent2])
                child2 = np.where(mask, population[parent2], population[parent1])
            else:
                raise ValueError("Invalid crossover choice")

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


def stopping_condition(fitness_scores, threshold):
    # Check if any of the fitness scores meet or exceed the stopping threshold
    return any(score >= threshold for score in fitness_scores)
