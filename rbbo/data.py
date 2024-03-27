import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomSmilesDataset(Dataset):
    def __init__(self, smiles_data):
        self.data = pd.read_csv(smiles_data)
        self.smiles = self.data.loc['smiles']
        self.target = self.data.loc['target']

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        target = self.target.iloc[idx]
        return smiles, target
