import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import glob

results_folder = "results"
df_experiment = pd.DataFrame()

ct = 0
algor = "ga_joblib"
for experiment_folder in os.listdir(results_folder):
    experiment_files = os.listdir(os.path.join(results_folder,experiment_folder))

    if len(experiment_files) != 3:
        continue
    
    for file in experiment_files:
        
        read_file = os.path.join(results_folder,experiment_folder,file)

        if "config" in file:
            df_config =pd.read_csv(read_file)

            # if df_config["algorithm"][0] == algor:
            if df_config["algorithm"][0] == algor and df_config["backend_prefer"][0] == "processes":
                print()
                print(df_config["model"][0])
                print(df_config["dataset"][0])
                print(os.path.join(results_folder,experiment_folder))
                ct += 1
print()
print(ct)