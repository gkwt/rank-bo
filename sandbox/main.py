import sys
import pandas as pd

sys.path.append("..")

from rbbo.data import MoleculeDataset
from rbbo.bo import BayesOptCampaign
from rbbo.models import MLP
from rbbo.metrics import frac_top_n, top_one

if __name__ == "__main__":
    dataset = MoleculeDataset("../data/delaney.csv", "fp", num_workers=1)
    bo = BayesOptCampaign(dataset, "maximize", "MLP", "fp", "mse", "greedy")
    bo_results = bo.run(1)
    # import pdb
    frac_top_10 = frac_top_n(dataset.data, bo_results[0], 10, "maximize")
    top_1 = top_one(bo_results[0])
    print(f"{frac_top_10=}, {top_1=}")

    # pdb.set_trace()
