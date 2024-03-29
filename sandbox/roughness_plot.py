import sys
sys.path.append('..')

from rogi import RoughnessIndex
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style="white")
import matplotlib.pyplot as plt

from rbbo.metrics import frac_top_n, top_one, custom_auc

if __name__ == "__main__":
    # input parameters
    goal = 'minimize'
    top_n = 20
    rough_dict = {"dataset": [], "loss_type": [], "auc": [], "rogi": [], }
    for dataset_name in ['CHEMBL2971_Ki_min_ROGI', 'CHEMBL4616_EC50_max_ROGI']:
        dataset_path = f'../data/{dataset_name}.csv'
        dataset: pd.DataFrame = pd.read_csv(dataset_path)
        ri = RoughnessIndex(Y=dataset["target"], smiles=dataset["smiles"])
        rogi_score = ri.compute_index()

        for loss in ["mse", "ranking"]:
            result_path = f'{dataset_name}_minimize/results_{loss}_{dataset_name}_minimize.pkl'
            results = pd.read_pickle(result_path)
            auc_list = []
            for i, df in enumerate(results):
                df['run'] = i
                df['loss_type'] = loss
                df = frac_top_n(dataset, df, top_n, goal)
                df = top_one(df, goal)
                auc = custom_auc(df, "top_one")
                auc_list.append(auc)

                rough_dict["dataset"].append(dataset_name)
                rough_dict["loss_type"].append(loss)
                rough_dict["auc"].append(auc)
                rough_dict["rogi"].append(rogi_score)
    rough_df = pd.DataFrame(rough_dict)
    
    sns.lineplot(x="rogi", y="auc", hue="loss_type",
                data=rough_df)
    plt.savefig(f'roughness_index_results_top_one.png')
    plt.close()