import sys
import pandas as pd

sys.path.append("..")

from rbbo.data import MoleculeDataset
from rbbo.bo import BayesOptCampaign
from rbbo.models import MLP

<<<<<<< Updated upstream

if __name__ == '__main__':

    dataset = MoleculeDataset(
        '../data/delaney.csv',
        'fp',
        num_workers=12
    )

    import pdb; pdb.set_trace()
=======
if __name__ == "__main__":
    model = MLP()
    dataset = MoleculeDataset("../data/delaney.csv", "fp", num_workers=1)
    bo = BayesOptCampaign(dataset, "maximize", model, "fp", "mse", "greedy")
    bo.run(1)
>>>>>>> Stashed changes
