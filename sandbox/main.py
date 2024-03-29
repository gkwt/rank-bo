import sys
sys.path.append("..")

import multiprocessing
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rbbo.data import MoleculeDataset
from rbbo.bo import BayesOptCampaign
from rbbo.models import MLP, get_loss_function

if __name__ == "__main__":
    # input parameters
    dataset_name = 'CHEMBL2971_Ki_min_ROGI'
    goal = 'minimize'
    loss_type = 'mse' # or ranking/mse

    num_workers = 20
    num_runs = 20
    work_dir = f'{dataset_name}_{goal}'

    dataset_path = f'../data/{dataset_name}.csv'
    dataset = MoleculeDataset(dataset_path, "fp", num_workers=num_workers)
    bo = BayesOptCampaign(dataset, goal, loss_type, verbose=False, work_dir=work_dir)

    # get best
    if goal == 'maximize':
        best_in_dataset = dataset.data.target.max()
    elif goal == 'minimize':
        best_in_dataset = dataset.data.target.min()

    # perform the run
    with multiprocessing.Pool(num_workers) as pool:
        bo_results = pool.map(bo.run, range(num_runs))

    pickle.dump(bo_results, open(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl', 'wb'))

