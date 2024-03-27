import torch
from torch.utils.data import Dataset

import multiprocessing

import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem

import mordred
import mordred.descriptors

calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)

def get_mordred_features(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return calc(m)._values

def get_fingerprint(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.GetMorganFingerprintAsBitVect(m, 3)


class MoleculeDataset(Dataset):
    def __init__(self, smiles_data: str, feature_type: str, num_workers: int = 1):
        self.data = pd.read_csv(smiles_data)
        self.smiles = self.data['smiles']
        self.target = self.data['target']
        self.feature_type = feature_type

        if feature_type == 'mordred':
            with multiprocessing.Pool(num_workers) as pool:
                desc = pool.map(get_mordred_features, self.smiles.tolist())
            self.data[feature_type] = desc
            desc = np.array(desc, dtype=float)

            self.data = self.data.dropna()
            self.feature = self.data[feature_type]
        elif feature_type == 'fp':
            with multiprocessing.Pool(num_workers) as pool:
                fps = pool.map(get_fingerprint, self.smiles.tolist())
            self.data[feature_type] = fps
            self.feature = self.data[feature_type]
        else:
            # if features is not defined, just use the smiles
            self.feature = self.smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        feature = self.feature.iloc[idx]
        target = self.target.iloc[idx]
        return feature, target
