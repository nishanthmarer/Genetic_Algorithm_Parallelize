import os
import re

import numpy as np
from subprocess import Popen
import pdb

import time
import numpy as np
import random

random.seed(123)
np.random.seed(123)

os.makedirs(os.path.join("logs"),exist_ok=True)

blocking=False
test_run=False

if test_run:
    models = ["mlp"]
    datasets = ["gina_agnostic"]
    algorithms = ["ga_spark"]
    metrics = ["f1"]
    crossovers = ["onepoint"]
    population_sizes = [10]
    elitisms = [2]
    mutations = [0.2]
    evolution_rounds = [10]
    backends = ["processes"]
    stopping_thresholds = [0.99999]

else:
    models = ["logistic","mlp","xgboost"]
    # models = ["mlp"]
    datasets = ["gina_agnostic","hiva_agnostic","sylva_agnostic"]
    # datasets = ["gina_agnostic","hiva_agnostic"]
    algorithms = ["baseline_metrics","ga_seq","ga_joblib","random","rfs","ga_spark"]
    # algorithms = ["ga_spark"]
    metrics = ["f1"]
    crossovers = ["onepoint"]
    population_sizes = [150]
    elitisms = [2]
    mutations = [0.2]
    evolution_rounds = [150]
    # backends = ["threads"]
    backends = ["processes"]
    stopping_thresholds = [0.99999]

if __name__ == "__main__":
    for back in backends:
        for popsize in population_sizes:
            for model in models:
                for data in datasets:
                    for algo in algorithms:
                        for metric in metrics:
                            for cross in crossovers:
                                for mut in mutations:
                                    for elit in elitisms:
                                        for evos in evolution_rounds:
                                            for stopping_threshold in stopping_thresholds:
                                                if algo != "ga_joblib" and back == "threads":
                                                    continue
                                                file = f"--population_size={popsize} "\
                                                f"--evolution_rounds={evos} "\
                                                f"--crossover_choice={cross} "\
                                                f"--metric_choice={metric} "\
                                                f"--elitism={elit} "\
                                                f"--dataset={data} "\
                                                f"--algorithm={algo} "\
                                                f"--model={model} "\
                                                f"--backend_prefer={back} "\
                                                f"--mutation_rate={mut} "\
                                                f"--stopping_threshold={stopping_threshold}"

                                                # print(file)
                                                if blocking:
                                                    file_full = f"python ga_fs.py {file}"
                                                    print(f'Running: {file_full}')
                                                    os.system(file_full)
                                                else:
                                                    file_full = f"python ga_fs.py {file}"
                                                    print(f"Running: sbatch execute.bash '{file_full}'")
                                                    Popen(f"sbatch execute.bash '{file_full}'",shell=True)

                                                time.sleep(0.01)
