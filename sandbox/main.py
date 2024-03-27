import sys
sys.path.append('..')

from rbbo.data import MoleculeDataset


if __name__ == '__main__':

    dataset = MoleculeDataset(
        '../data/delaney.csv',
        'fp',
        num_workers=12
    )