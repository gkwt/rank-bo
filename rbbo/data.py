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
    return np.array(Chem.GetMorganFingerprintAsBitVect(m, 3), dtype=float)


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

    def __getitem__(self, idx: int):
        feature = self.data["feature"].iloc[idx]
        target = self.data["target"].iloc[idx]
        return feature, target


class DataframeDataset(Dataset):
    """
    Quick wrapper to create a dataset for pytorch training
    directly from dataframe.
    Requires a column named "feature" and one named "target"
    """

    def __init__(self, df: pd.DataFrame):
        self.data = df
        assert "feature" in df.columns, 'Column for "feature" not found.'
        assert "target" in df.columns, 'Column for "target" not found.'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data["feature"].iloc[idx], self.data["target"].iloc[idx]

class PairwiseRankingDataframeDataset(Dataset):
    """
    This dataset will load the data for pairwise ranking loss.
    """
    def __init__(self, df: pd.DataFrame, max_num_pairs: int = 0):
        self.data = df
        assert "feature" in df.columns, 'Column for "feature" not found.'
        assert "target" in df.columns, 'Column for "target" not found.'

        self.max_num_pairs = max_num_pairs

        # default to 2*length of dataframe
        if self.max_num_pairs == 0:
            self.max_num_pairs = 2*len(df)

        if self.max_num_pairs > len(df)**2:
            self.max_num_pairs = len(df)
         
        # the ranking based on the target value
        self.compare_fn = np.greater

        # get indices for pairs
        # will produce (n^2-n)/2 data pairs, which can be truncated
        pairs = np.array(np.triu_indices(len(df), k=1, m=len(df))).transpose()
        
        if max_num_pairs >= 0:
            self.pairs = pairs[np.random.choice(pairs.shape[0], self.max_num_pairs, replace=False), :]
        else:
            self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # m(x1) > m(x2) is y = +1
        # m(x1) < m(x2) is y = -1
        idx = self.pairs[idx]
        target = self.compare_fn(self.data.iloc[idx[0]].target, self.data.iloc[idx[1]].target).astype(float)
        target = target * 2.0 - 1.0

        return self.data.iloc[idx[0]]['feature'], self.data.iloc[idx[1]]['feature'], target



    



