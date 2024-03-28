import sys
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")

from rbbo.data import MoleculeDataset
from rbbo.bo import BayesOptCampaign
from rbbo.models import MLP, get_loss_function
from rbbo.metrics import frac_top_n, top_one

if __name__ == "__main__":
    dataset_name = 'delaney'
    num_workers = 20
    num_runs = 20
    loss_type = 'ranking' # or mse

    dataset_path = f'../data/{dataset_name}.csv'
    dataset = MoleculeDataset(dataset_path, "fp", num_workers=num_workers)
    bo = BayesOptCampaign(dataset, "maximize", loss_type)

    with multiprocessing.Pool(num_workers) as pool:
        bo_results = pool.map(bo.run, range(num_runs))

    # calculate the statistics
    for i, df in enumerate(bo_results):
        df['run'] = i
        df = frac_top_n(dataset.data, df, 20, 'maximize')
        df = top_one(df)

    df_all = pd.concat(bo_results).reset_index()
    df_all.to_csv(f'results_{loss_type}_{dataset_name}.csv', index=False)
