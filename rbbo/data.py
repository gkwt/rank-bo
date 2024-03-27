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
    """
    Load library of pytorch dataset. Specify the feature type.
    Featurizes the dataset.
    """

    def __init__(self, smiles_data: str, feature_type: str, num_workers: int = 1):
        self.data = pd.read_csv(smiles_data)
        self.smiles = self.data["smiles"]
        self.target = self.data["target"]
        self.feature_type = feature_type

        if feature_type == "mordred":
            with multiprocessing.Pool(num_workers) as pool:
                desc = pool.map(get_mordred_features, self.smiles.tolist())
            desc = np.array(desc, dtype=float)
            desc = desc[:, ~np.isnan(desc).any(axis=0)]

            self.data["feature"] = list(desc)
            self.data = self.data.dropna(axis=1)
            self.feature = self.data["feature"]

        elif feature_type == "fp":
            with multiprocessing.Pool(num_workers) as pool:
                fps = pool.map(get_fingerprint, self.smiles.tolist())
            self.data["feature"] = fps
            self.feature = self.data["feature"]

        else:
            # if features is not defined, just use the smiles
            self.feature = self.smiles

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data["feature"].iloc[idx]
        target = self.data["target"].iloc[idx]
        return feature, target


class DataframeDataset(Dataset):
    """
    Quick wrapper to create a dataset for pytorch training
    directly from dataframe.
    Requires a column named "feature" and one named "target"
    """

    def __init__(self, df):
        self.data = df
        assert "feature" in df.columns, 'Column for "feature" not found.'
        assert "target" in df.columns, 'Column for "target" not found.'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data["feature"].iloc[idx], self.data["target"].iloc[idx]
