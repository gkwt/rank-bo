import sys
sys.path.append('..')

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rbbo.metrics import frac_top_n, top_one, frac_top_n_percent

def read_bo_pickle(filename):
    return pickle.load(open(filename, 'rb'))

if __name__ == '__main__':
    dataset_name = 'CHEMBL4616_EC50_max_ROGI'
    goal = 'minimize'
    top_n = 10

    work_dir = f'{dataset_name}_{goal}'

    # get dataset
    dataset_df = pd.read_csv(f'../data/{dataset_name}.csv')
    best_in_dataset = dataset_df.target.min() if goal == 'minimize' else dataset_df.target.max()

    # get the optimizations
    df_all = []
    for loss_type in ['mse', 'ranking', 'random']:
        if loss_type == 'random':
            # make results for a random optimization
            bo_results = []
            for _ in range(n_runs):
                np.random.seed(_)
                bo_results.append(dataset_df.sample(n_evals, ignore_index=True).reset_index())
        else:
            bo_results = read_bo_pickle(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl')
            n_runs = len(bo_results)
            n_evals = len(bo_results[0])

        for i, df in enumerate(bo_results):
            df['run'] = i
            df['loss_type'] = loss_type
            # df = frac_top_n(dataset_df, df, top_n, goal)
            df = frac_top_n_percent(dataset_df, df, top_n, goal)
            df = top_one(df, goal)
        
        # combine the dataset
        bo_results = pd.concat(bo_results).reset_index()
        df_all.append(bo_results)

    df_all = pd.concat(df_all, ignore_index=True)

    sns.lineplot(df_all, x='index', y='top_one', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.xlabel('Evaluations')
    plt.ylabel(f'Top1 fitness')
    plt.xlim([0,n_evals-1])
    plt.hlines(best_in_dataset, 0, n_evals, 'k', '--')
    plt.ylim([0,2]) # you may have to change this depending on the analysis
    plt.savefig(f'{dataset_name}_{goal}/{dataset_name}_top1.png')
    plt.close()

    sns.lineplot(df_all, x='index', y='frac_top_n_percent', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.xlabel('Evaluations')
    plt.ylabel(f'Fraction of top {top_n}% of dataset')
    plt.xlim([0,n_evals-1])
    plt.ylim([0,1])
    plt.savefig(f'{dataset_name}_{goal}/{dataset_name}_fracs{top_n}.png')
    plt.close()

