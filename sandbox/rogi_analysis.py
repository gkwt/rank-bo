import sys
WORK_DIR = '..'
sys.path.append(WORK_DIR)

from rogi import RoughnessIndex
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="white")
sns.set_palette("colorblind")

import matplotlib.pyplot as plt

from rbbo.metrics import frac_top_n, top_one, auc_metric
from rbbo.utils import ORACLE_NAMES, ORACLE_OBJ

if __name__ == "__main__":
    # input parameters
    # goal = 'minimize'
    model_type='bnn'
    acq_func='ei'
    top_n = 100
    rough_dict = {"dataset": [], "loss_type": [], "auc": [], "rogi": [], }

    for dataset_name in ORACLE_NAMES:
        goal = ORACLE_OBJ[dataset_name]
        dataset_name = 'zinc_' + dataset_name
        dataset_path = f'{WORK_DIR}/data/{dataset_name}.csv'
        dataset: pd.DataFrame = pd.read_csv(dataset_path)

        ri = RoughnessIndex(Y=dataset["target"], smiles=dataset["smiles"])
        rogi_score = ri.compute_index()

        for loss in ["mse", "ranking"]:
            # result_path = f'rogi_{model_type}_{acq_func}/{dataset_name}_{goal}_{model_type}_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
            result_path = f'{dataset_name}_{goal}_{model_type}_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
            results = pd.read_pickle(result_path)
            for i, res in enumerate(results):
                if type(res) is tuple:
                    df, preds = res
                else:
                    df = res
                df['run'] = i
                df['loss_type'] = loss
                df = frac_top_n(dataset, df, top_n, goal)
                df = top_one(df, goal)

                auc = auc_metric(
                    dataset, 
                    bo_output = df, 
                    metric = "frac_top_n", 
                    goal = goal
                )

                rough_dict["dataset"].append(dataset_name)
                rough_dict["loss_type"].append(loss)
                rough_dict["auc"].append(auc)
                rough_dict["rogi"].append(rogi_score)
        

        loss = "mse"
        result_path = f'rogi_gp_{acq_func}/{dataset_name}_{goal}_gp_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
        result_path = f'{dataset_name}_{goal}_gp_{acq_func}/results_{loss}_{dataset_name}_{goal}.pkl'
        results = pd.read_pickle(result_path)

        for i, res in enumerate(results):
            if type(res) is tuple:
                df, preds = res
            else:
                df = res
            df['run'] = i
            df['loss_type'] = loss
            df = frac_top_n(dataset, df, top_n, goal)
            df = top_one(df, goal)

            auc = auc_metric(
                dataset, 
                bo_output = df, 
                metric = "frac_top_n", 
                goal = goal
            )

            rough_dict["dataset"].append(dataset_name)
            rough_dict["loss_type"].append("gp + mll")
            rough_dict["auc"].append(auc)
            rough_dict["rogi"].append(rogi_score)

    rough_df = pd.DataFrame(rough_dict)
    
    sns.lineplot(rough_df, x="rogi", y="auc", hue="loss_type", marker='o', linestyle='', err_style='bars', hue_order=['mse', 'ranking', 'gp + mll'], alpha=0.95)
    # plt.ylim([0, 0.25])        # will need to adjust based on your runs
    plt.savefig(f'rogi_auc_{model_type}_{acq_func}.png', bbox_inches='tight')
    plt.close()