import sys
sys.path.append('..')

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, kendalltau

from argparse import ArgumentParser

from rbbo.bo import BayesOptCampaign
from rbbo.metrics import frac_top_n, top_one, frac_top_n_percent


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", action="store", type=str, help="Protein target, defaults 1OYT.")
    parser.add_argument("--maximize", action="store_true", dest="goal", help="Set goal to maximize. Otherwise will minimize.", default=False)
    parser.add_argument("--top_n", action="store", type=int, help="Fraction of top n found. Defaults 100.", default=100)
    parser.add_argument("--top_percentile", action="store", type=float, help="Fraction of mols in top percentile found. Defaults 10 percent.", default=10.0)
    parser.add_argument("--model_type", action="store", type=str, help="Model type. Defaults mlp.", default='mlp')
    parser.add_argument("--acq_func", action="store", type=str, help="Acquisition function. Defaults to greedy.", default='greedy')
    FLAGS = parser.parse_args()

    dataset_name = FLAGS.dataset_name
    goal = 'maximize' if FLAGS.goal else 'minimize'
    top_n = FLAGS.top_n
    top_percentile = FLAGS.top_percentile

    work_dir = f'{dataset_name}_{goal}_{FLAGS.model_type}_{FLAGS.acq_func}'
    
    # get dataset
    dataset_df = pd.read_csv(f'../data/{dataset_name}.csv')
    best_in_dataset = dataset_df.target.min() if goal == 'minimize' else dataset_df.target.max()

    # get the optimizations
    df_all, pred_all = [], []
    for loss_type in ['mse', 'ranking', 'random']:
        if loss_type == 'random':
            # make results for a random optimization
            bo_results = []
            for seed in range(n_runs):
                np.random.seed(seed)
                bo_run = []
                for _ in range(n_evals):
                    bo_run.append(BayesOptCampaign.sample_meas_randomly(dataset_df))
                bo_results.append(
                    pd.concat(bo_run, axis=1, ignore_index=True).transpose().reset_index().rename(columns={'index': 'evaluation'})
                )
        else:
            try:
                bo_results = pd.read_pickle(f'{work_dir}/results_{loss_type}_{dataset_name}_{goal}.pkl')
                n_runs = len(bo_results)
                n_evals = len(bo_results[0][0])
            except:
                print(f'Missing {loss_type} run.')
                continue

        df_collect = []
        pred_collect = []
        for i, res in enumerate(bo_results):
            if type(res) is tuple:
                # calculate the performance metrics
                df, preds = res
                for j, gdf in preds.groupby('iteration'):
                    r2 = r2_score(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                    p_rho, _ = pearsonr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                    s_rho, _ = spearmanr(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                    tau, _ = kendalltau(gdf['y_true'].tolist(), gdf['y_pred'].tolist())
                    pred_collect.append({
                        'iteration': j, 
                        'loss_type': loss_type,
                        'r2': r2, 
                        'spearman': s_rho,
                        'pearson': p_rho,
                        'kendall': tau,
                    })
            else:
                df = res

            df['run'] = i
            df['loss_type'] = loss_type
            df = frac_top_n(dataset_df, df, top_n, goal)
            df = frac_top_n_percent(dataset_df, df, top_percentile, goal)
            df = top_one(df, goal)
            df_collect.append(df)
        
        # combine the dataset
        df_all.append(pd.concat(df_collect))
        pred_all.append(pd.DataFrame(pred_collect))

    df_all = pd.concat(df_all)
    pred_all = pd.concat(pred_all)

    # plot the analysis
    sns.lineplot(df_all, x='evaluation', y='top_one', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.xlabel('Evaluations')
    plt.ylabel(f'Top1 fitness')
    plt.xlim([0,n_evals-1])
    plt.hlines(best_in_dataset, 0, n_evals, 'k', '--')
    # plt.ylim([0,2]) # you may have to change this depending on the analysis
    plt.savefig(f'{work_dir}/{dataset_name}_top1.png', bbox_inches='tight')
    plt.close()

    sns.lineplot(df_all, x='evaluation', y='frac_top_n', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.xlabel('Evaluations')
    plt.ylabel(f'Fraction of top {top_n} of dataset')
    plt.xlim([0,n_evals-1])
    # plt.ylim([0,1])
    plt.savefig(f'{work_dir}/{dataset_name}_fracs{top_n}.png', bbox_inches='tight')
    plt.close()

    sns.lineplot(df_all, x='evaluation', y='frac_top_n_percent', hue='loss_type', hue_order=['mse', 'ranking', 'random'])
    plt.xlabel('Evaluations')
    plt.ylabel(f'Fraction of top {top_percentile}% of dataset')
    plt.xlim([0,n_evals-1])
    # plt.ylim([0,1])
    plt.savefig(f'{work_dir}/{dataset_name}_percentile{top_percentile}.png', bbox_inches='tight')
    plt.close()

    # performance metrics
    sns.lineplot(pred_all, x='iteration', y='r2', hue='loss_type', hue_order=['mse', 'ranking'])
    plt.xlabel('Iteration')
    plt.ylabel(f'R2 score')
    plt.xlim([0,pred_all['iteration'].max()])
    plt.savefig(f'{work_dir}/{dataset_name}_r2.png', bbox_inches='tight')
    plt.close()

    sns.lineplot(pred_all, x='iteration', y='kendall', hue='loss_type', hue_order=['mse', 'ranking'])
    plt.xlabel('Iteration')
    plt.ylabel(f'Kendall tau correlation')
    plt.xlim([0,pred_all['iteration'].max()])
    plt.savefig(f'{work_dir}/{dataset_name}_kendall.png', bbox_inches='tight')
    plt.close()

    sns.lineplot(pred_all, x='iteration', y='spearman', hue='loss_type', hue_order=['mse', 'ranking'])
    plt.xlabel('Iteration')
    plt.ylabel(f'Spearman correlation')
    plt.xlim([0,pred_all['iteration'].max()])
    plt.savefig(f'{work_dir}/{dataset_name}_spearman.png', bbox_inches='tight')
    plt.close()

