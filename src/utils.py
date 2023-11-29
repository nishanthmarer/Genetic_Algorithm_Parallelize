from sklearn.datasets import load_breast_cancer,fetch_20newsgroups_vectorized,fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

import pandas as pd
def load_dataset(dataset):
    os.makedirs("dataset_cache", exist_ok=True)

    X, y = fetch_openml(dataset,return_X_y=True,as_frame=True,data_home="dataset_cache")

    if dataset=="SantanderCustomerSatisfaction":
        X.drop(["ID_code"],inplace=True,axis=1)

    X = pd.get_dummies(X,columns=X.columns[X.dtypes == "object"])

    label_encoder = LabelEncoder()

    X = X.values.astype(float); y=label_encoder.fit_transform(y).astype(float)
    print("Dataset Shape: ",X.shape)

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

    return X_tr,X_te,y_tr,y_te