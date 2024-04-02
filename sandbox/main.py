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

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", action="store", type=str, help="Protein target, defaults 1OYT.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--num_runs", action="store", type=int, dest="num_runs", help="Number of runs in BO. Defaults 20.", default=20)
    parser.add_argument("--num_init", action="store", type=int, dest="num_init", help="Number of initial samples defaults 20.", default=20)
    parser.add_argument("--maximize", action="store_true", dest="goal", help="Set goal to maximize. Otherwise will minimize.", default=False)
    parser.add_argument("--rank", action="store_true", help="Toggle use of ranking loss. Otherwise, MSE.", default=False)
    FLAGS = parser.parse_args()

    # input parameters
    dataset_name = FLAGS.dataset_name
    goal = 'maximize' if FLAGS.maximize else 'minimize'
    loss_type = 'rank' if FLAGS.rank else 'mse' 
    num_workers = FLAGS.num_workers
    num_runs = FLAGS.num_runs
    num_init = FLAGS.num_init
    work_dir = f'{dataset_name}_{goal}'

    dataset_path = f'../data/{dataset_name}.csv'
    dataset = MoleculeDataset(dataset_path, "fp", num_workers=num_workers)
    bo = BayesOptCampaign(
        dataset, 
        goal, 
        loss_type, 
        budget=num_runs, 
        num_init_design = num_init, 
        verbose=False, 
        work_dir=work_dir
    )

    # perform the run
    with multiprocessing.Pool(num_workers) as pool:
        bo_results = pool.map(bo.run, range(num_runs))

    pickle.dump(bo_results, open(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl', 'wb'))

