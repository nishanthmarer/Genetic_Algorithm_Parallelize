from sklearn.datasets import load_breast_cancer,fetch_20newsgroups_vectorized,fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import numpy as np

import os

import pandas as pd
def load_dataset(dataset,test_size=0.1):
    os.makedirs("dataset_cache", exist_ok=True)
    label_encoder = LabelEncoder()

    if dataset in ["IMDB.drama"]:
        X, y = fetch_openml(dataset, return_X_y=True, as_frame=False)#, data_home="dataset_cache")
        X = X.astype(float); y=label_encoder.fit_transform(y).astype(float)

    if dataset == "sylva_agnostic":
        X, y = fetch_openml(dataset,return_X_y=True,as_frame=True)#,data_home="dataset_cache")
        y = X.pop("label")
        X = pd.get_dummies(X,columns=X.columns[(X.dtypes == "object") | (X.dtypes == "category")])
        X = X.values.astype(float); y=label_encoder.fit_transform(y).astype(float)

    else:
        X, y = fetch_openml(dataset,return_X_y=True,as_frame=True)#,data_home="dataset_cache")

        if dataset=="SantanderCustomerSatisfaction":
            X.drop(["ID_code"],inplace=True,axis=1)

        X = pd.get_dummies(X,columns=X.columns[(X.dtypes == "object") | (X.dtypes == "category")])

        X = X.values.astype(float); y=label_encoder.fit_transform(y).astype(float)
    print("Dataset Shape: ",X.shape)

    val_size = 0.2/(1-test_size)
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=test_size,stratify=y,random_state=123)
    X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=val_size,stratify=y_tr,random_state=123)

    std_scaler = StandardScaler()

    X_tr = std_scaler.fit_transform(X_tr)
    X_val= std_scaler.transform(X_val)
    X_te = std_scaler.transform(X_te)

    return X_tr,X_val,X_te,y_tr,y_val,y_te

def select_model(model):
    if model == "logistic":
        clf = LogisticRegression
    if model == "mlp":
        clf = MLPClassifier
    if model == "xgboost":
        clf = XGBClassifier

    return clf

def csv_writer_util(filename,evo,total_time,scores,best_chromosome):
    if os.path.isfile(filename):
        data = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)],
                             'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
        data.to_csv(filename, mode='a', header=False, index=False)
    else:
        columns = pd.DataFrame(columns=['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'])
        result = pd.DataFrame({'Gen': [evo], 'Time': [total_time], 'Avg': [np.mean(scores)], 'Best': [np.max(scores)],
                               'Worst': [np.min(scores)], 'Best Chromosome': [best_chromosome]})
        data = pd.concat((columns, result))
        data.to_csv(filename, header=['Gen', 'Time', 'Avg', 'Best', 'Worst', 'Best Chromosome'], index=False)