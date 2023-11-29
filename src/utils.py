from sklearn.datasets import load_breast_cancer,fetch_20newsgroups_vectorized,fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
def load_dataset(dataset):
    X, y = fetch_openml(dataset,return_X_y=True,as_frame=True)

    if dataset=="SantanderCustomerSatisfaction":
        X.drop(["ID_code"],inplace=True,axis=1)

    X = pd.get_dummies(X,columns=X.columns[X.dtypes == "object"])

    X = X.values.astype(float); y=y.values.astype(float)
    print("Dataset Shape: ",X.shape)

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

    return X_tr,X_te,y_tr,y_te