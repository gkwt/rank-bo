import sys
sys.path.append('..')

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rbbo.metrics import frac_top_n, top_one


def read_bo_pickle(filename):
    return pickle.load(open(filename, 'rb'))


if __name__ == '__main__':
    dataset_name = 'CHEMBL2971_Ki_min_ROGI'
    goal = 'minimize'
    top_n = 20

    work_dir = f'{dataset_name}_{goal}'

    # get dataset
    dataset_df = pd.read_csv(f'../data/{dataset_name}.csv')
    best_in_dataset = dataset_df.target.min() if goal == 'minimize' else dataset_df.target.max()

    # get the optimizations
    df_all = []
    for loss_type in ['mse', 'random']: #['mse','ranking', 'random']:
        if loss_type == 'random':
            # make results for a random optimization
            bo_results = [dataset_df.sample(n_evals, ignore_index=True).reset_index() for _ in range(n_runs)]
        else:
            bo_results = read_bo_pickle(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl')
            n_runs = len(bo_results)
            n_evals = len(bo_results[0])

        for i, df in enumerate(bo_results):
            df['run'] = i
            df['loss_type'] = loss_type
            df = frac_top_n(dataset_df, df, top_n, goal)
            df = top_one(df, goal)
        
        # combine the dataset
        import pdb; pdb.set_trace()
        bo_results = pd.concat(bo_results).reset_index()
        df_all.append(bo_results)

    df_all = pd.concat(df_all, ignore_index=True)

    sns.lineplot(df_all, x='index', y='top1', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.ylim([0,2]) # you may have to change this depending on the analysis
    plt.savefig(f'{dataset_name}_{goal}/{dataset_name}_top1.png')
    plt.close()

    sns.lineplot(df_all, x='index', y='fracs', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.ylim([0,1])
    plt.savefig(f'{dataset_name}_{goal}/{dataset_name}_fracs{top_n}.png')
    plt.close()

