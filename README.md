# Genetic_Algorithm_Parallelize
Parallelize the Genetic Algorithm (GA) for feature selection

Datasets may be found at
1. [OpenML](https://www.openml.org/search?type=data&status=active&qualities.NumberOfClasses=%3D_2&id=4134)
2. [UC Irvine MLR](https://archive.ics.uci.edu/datasets)

Where OpenML is compatible with `sklearn.datasets.fetch_openml` to download and process datasets into `numpy` arrays compatible with `sklearn` apis.

On the cluser the following commands need to be executed before the project can be run:

1. 'module load anaconda3/2022.05'
2. 'conda activate <env>'