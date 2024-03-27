import sys
import pandas as pd

sys.path.append("..")

from rbbo.data import MoleculeDataset
from rbbo.bo import BayesOptCampaign
from rbbo.models import MLP

if __name__ == "__main__":
    dataset = MoleculeDataset("../data/delaney.csv", "fp", num_workers=1)
    bo = BayesOptCampaign(dataset, "maximize", "MLP", "fp", "mse", "greedy")
    bo.run(1)
    # import pdb

    # pdb.set_trace()
