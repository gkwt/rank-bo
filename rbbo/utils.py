import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

ORACLE_NAMES = ['QED', 'LogP', 'Celecoxib_Rediscovery', 'Aripiprazole_Similarity', 'Median 1', 
                'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', 'Amlodipine_MPO', 'Zaleplon_MPO',
                'Scaffold Hop']

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
