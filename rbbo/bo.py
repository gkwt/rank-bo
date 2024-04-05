import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss

import numpy as np
import pandas as pd

from rbbo.data import DataframeDataset, PairwiseRankingDataframeDataset
from rbbo.models import MLP, BNN, get_loss_function


class BayesOptCampaign:
    """
    Args:
        dataset (pd.DataFrame)
        goal (str): optimization goal, 'maximize' or 'minimize'
        model
        acq_func_type (str): acquisition function type
        num_acq_samples (int): number of samples drawn in each round of acquisition function optimization
        batch_size (int): number of recommendations provided by the acquisition function at each
            iteration
        budget (int): maximum tolerated objective function measurements for a single
            optimization campaign
        init_design_strategy (str): strategy used to select the initial design points
        num_init_design (int): number of initial design points to propose
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        goal: str,
        loss_type: str,
        acq_func: Callable,
        model_type: str = 'mlp',
        num_of_epochs: int = 5,
        num_total: int = 10,
        batch_size: int = 1,
        budget: int = 100,
        init_design_strategy: str = "random",
        num_init_design: int = 20,
        work_dir: str = ".",
        verbose: bool = True,
        num_acq_samples: int = -1,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.goal = goal
        self.model_type = model_type
        self.acq_func = acq_func
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir
        self.num_init_design = num_init_design
        self.loss_type = loss_type
        self.loss_func = get_loss_function(loss_type)
        self.num_of_epochs = num_of_epochs
        self.num_total = num_total
        self.verbose = verbose
        self.num_acq_samples = num_acq_samples

        if use_gpu:
            if torch.cuda.is_available():
                print('GPU found, and is used.')
                self.use_gpu = True
                self.device = torch.device('cuda')
            else:
                print('No GPU found, default to CPU.')
                self.use_gpu = False
                self.device = torch.device('cpu')
        else:
            self.use_gpu = False

        # create working dir, and write the hypermarameters
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.work_dir + '/hparams.txt', 'w') as f: 
            for k,v in self.__dict__.items():
                f.write(f'{k}: {v}\n')

    def _validate_budget(self):
        """validate that the budget does not exceed the total number of
        options in the dataset
        """
        if self.budget > self.num_total:
            print(
                f"Requested budget exceeeds the number of total candidates. Resetting to {self.num_total}"
            )
            self.budget = self.num_total - self.num_init_design

    def _reinitialize_model(self):
        """Reinitialize the model from scratch (re-train)"""
        if self.model_type == 'mlp':
            reinit_model = MLP().double()
        elif self.model_type == 'bnn':
            reinit_model = BNN().double()
        return reinit_model

    def run(self, seed: int = None):
        """Run the sequential learning experiments for num_restart independently seeded
        executions.
        """

        # set seed for reproducibility
        if not seed:
            np.random.seed(seed)

        observations = []  # [ {'smiles': ,'target': ,'feature': }, ..., ]
        iter_num = 0

        while (len(observations) - self.num_init_design) < self.budget:
            # re-initialize the surrogate model
            self.model = self._reinitialize_model()
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.001,
            )

            if iter_num == 0:
                for _ in range(self.num_init_design):
                    sample, measurement, feature = self.sample_meas_randomly(
                        self.dataset.data
                    )
                    observations.append(
                        {
                            "smiles": sample,
                            "target": measurement,
                            "feature": feature,
                        }
                    )

            # split dataset into measured and available candidates
            meas_df, avail_df = self.split_avail(self.dataset.data, observations)
            # shuffle the available candidates (acq func sampling)
            if self.num_acq_samples > 0 and len(avail_df) > self.num_acq_samples:
                avail_df = avail_df.sample(n=self.num_acq_samples).reset_index(drop=True)
            if self.verbose:
                print(
                    f"NUM_ITER : {iter_num}\tNUM OBS : {len(observations)}"
                )

            # convert X_meas and y_meas to torch Dataset
            if self.loss_type == 'mse':
                meas_set = DataframeDataset(meas_df)
            elif self.loss_type == 'ranking':
                meas_set = PairwiseRankingDataframeDataset(meas_df)
            avail_set = DataframeDataset(avail_df)

            # load data use DataLoader
            meas_loader = torch.utils.data.DataLoader(
                meas_set, batch_size=8, shuffle=True, drop_last=False
            )
            avail_loader = torch.utils.data.DataLoader(
                avail_set, batch_size=128, shuffle=False
            )

            # train the model on observations
            # start with fresh model every time
            # specify the LOSS function -> (ranking/mse)
            for epoch in range(self.num_of_epochs):
                # train the model
                self.model.train()

                for data in meas_loader:
                    self.optimizer.zero_grad()
                    loss = self.model.train_step(
                        data, 
                        loss_type=self.loss_type, 
                        loss_func=self.loss_func,
                        device=self.device
                    )

                    loss.backward()
                    self.optimizer.step()

            # make inference
            mu_avail, std_avail = [], []
            with torch.no_grad():
                self.model.eval()
                for data in avail_loader:
                    X_avail, _ = data
                    X_avail = X_avail.to(self.device)
                    y_avail, y_std = self.model.predict(X_avail)
                    mu_avail.append(y_avail.detach().cpu().numpy())
                    if y_std is not None:
                        std_avail.append(y_std.detach().cpu().numpy())

            mu_avail = np.concatenate(mu_avail).flatten()
            if not std_avail:
                std_avail = None
            else:
                std_avail = np.concatenate(std_avail).flatten()

            # calculate acq function
            # negate the results of prediction if minimizing
            if self.goal == "minimize":
                mu_avail *= -1.0
                y_best = -meas_df['target'].min()    # negate and calculate the max
            elif self.goal == "maximize":
                y_best = meas_df['target'].max()
            else:
                raise ValueError('Goal must be minimize or maximize.')

            acq_vals = self.acq_func(mu_avail, std_avail, best_val=y_best)   
            
            # higher acq_vals the better
            sort_idxs = np.argsort(acq_vals)[::-1]  # descending order
            sample_idxs = sort_idxs[: self.batch_size]

            # perform measurements
            for sample_idx in sample_idxs:
                sample, measurement, feature = self.sample_meas_acq(
                    avail_df, sample_idx
                )
                observations.append(
                    {"smiles": sample, "target": measurement, "feature": feature}
                )

            iter_num += 1

        return pd.DataFrame(observations)

    @staticmethod
    def sample_meas_acq(avail_df, idx):
        """obtain the molecules suggested by the acquisition function"""
        return avail_df.iloc[idx]

    @staticmethod
    def sample_meas_randomly(avail_df):
        """take a single random sample from the available candiates"""
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx]

    @staticmethod
    def split_avail(data, observations):
        """return available and measured datasets"""
        obs_smi = [o["smiles"] for o in observations]

        # avail_df is the set of molecules that have not been measured
        # create a function that checks if the smiles is in
        avail_df = data[~(data["smiles"].isin(obs_smi))]
        meas_df = data[data["smiles"].isin(obs_smi)]
        return meas_df, avail_df

