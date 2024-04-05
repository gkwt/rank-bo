import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

ORACLE_NAMES = ['QED', 'LogP', 'Celecoxib_Rediscovery', 'Aripiprazole_Similarity', 'Median_1', 
                'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', 'Amlodipine_MPO', 'Zaleplon_MPO',
                'Scaffold_Hop']

ORACLE_OBJ = {
    'QED': 'maximize',
    'LogP': 'minimize',
    'Celecoxib_Rediscovery': 'maximize', 
    'Aripiprazole_Similarity': 'maximize', 
    'Median_1': 'maximize', 
    'Osimertinib_MPO': 'maximize', 
    'Fexofenadine_MPO': 'maximize', 
    'Ranolazine_MPO': 'maximize', 
    'Perindopril_MPO': 'maximize', 
    'Amlodipine_MPO': 'maximize', 
    'Zaleplon_MPO': 'maximize',
    'Scaffold_Hop': 'maximize'
}

def min_max_scale(x: pd.Series, min_val: float = None, max_val: float = None):
    if not min_val or not max_val:
        return minmax_scale(x)
    scaled_x = (x - min_val) / (max_val - min_val)
    return scaled_x

def remove_outliers(df: pd.DataFrame, goal: str = 'maximize', num_sigma: float = 3.0):
    mu = df['target'].mean()
    std = df['target'].std()

    if goal == 'maximize':
        df = df[df['target'] > mu - num_sigma*std]
    elif goal == 'minimize':
        df = df[df['target'] < mu + num_sigma*std]
    return df

def get_split_indices(num_runs, n_procs):
    indices = list(range(num_runs))
    chunk_size, remaining = divmod(num_runs, n_procs)
    split_inds = []
    for i in range(n_procs):
        start = i * chunk_size + min(i, remaining)
        end = (i + 1) * chunk_size + min(i + 1, remaining)
        split_inds.append(indices[start:end])
    return split_inds
