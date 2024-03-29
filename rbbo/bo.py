import os

import torch
import torch.nn as nn
from torch import optim
from torch.nn import MSELoss

import numpy as np
import pandas as pd

from rbbo.data import DataframeDataset, PairwiseRankingDataframeDataset
from rbbo.models import MLP, get_loss_function


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
        loss_func: str,
        num_of_epochs: int = 5,
        num_total: int = 10,
        # num_acq_samples: int = 50,
        batch_size: int = 1,
        budget: int = 100,
        init_design_strategy: str = "random",
        num_init_design: int = 20,
        work_dir: str = ".",
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.goal = goal
        # self.model = model
        # self.feature_type = feature_type
        # self.acq_func_type = acq_func_type
        # self.num_acq_samples = num_acq_samples
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir
        self.num_init_design = num_init_design
        self.loss_type = loss_func
        self.loss_func = get_loss_function(loss_func)
        # self.num_workers = num_workers
        self.num_of_epochs = num_of_epochs
        self.num_total = num_total
        self.verbose = verbose

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
        reinit_model = MLP().double()
        return reinit_model

    def run(self, seed: int = None):
        """Run the sequential learning experiments for num_restart independently seeded
        executions.
        """
        if seed:
            np.random.seed(seed)

        observations = []  # [ {'smiles': ,'target': ,'feature': }, ..., ]
        all_test_set_metrics = []
        iter_num = 0

        while (len(observations) - self.num_init_design) < self.budget:
            # re-initialize the surrogate model
            self.model = self._reinitialize_model()
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
            meas_df, avail_df = self.split_avail(self.dataset, observations)
            # shuffle the available candidates (acq func sampling)
            avail_df = avail_df.sample(frac=1).reset_index(drop=True)
            if self.verbose:
                print(
                    f"NUM_ITER : {iter_num}\tNUM OBS : {len(observations)}"
                )
            # elif self.model_type == "nn":
            #     # use nearnest neighbour strategy
            #     X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)
            #     if self.goal == "minimize":
            #         best_fp = np.array(
            #             meas_df.nsmallest(n=1, columns="target")["feature"].tolist()
            #         ).squeeze()
            #     elif self.goal == "maximize":
            #         best_fp = np.array(
            #             meas_df.nlargest(n=1, columns="target")["feature"].tolist()
            #         ).squeeze()
            #     sims = np.array(
            #         [chem.similarity_between_fps(x, best_fp) for x in X_avail]
            #     )
            #     ind = np.argsort(sims)[-self.batch_size :]
            #     for sample_idx in ind:
            #         sample, measurement = self.sample_meas_acq(
            #             avail_df, sample_idx
            #         )
            #     observations.append({"smi": sample, "target": measurement})

            # sample using surrogate model and acquisition
            # X_meas, y_meas = self.make_xy(meas_df)
            # X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)

            # convert X_meas and y_meas to torch Dataset
            if self.loss_type == 'mse':
                meas_set = DataframeDataset (meas_df)
            elif self.loss_type == 'ranking':
                meas_set = PairwiseRankingDataframeDataset(meas_df)
            avail_set = DataframeDataset(avail_df)

            # load data use DataLoader
            meas_loader = torch.utils.data.DataLoader(
                meas_set, batch_size=8, shuffle=True, drop_last=True
            )
            avail_loader = torch.utils.data.DataLoader(
                avail_set, batch_size=64, shuffle=False
            )

            # train the model on observations
            # start with fresh model every time
            # specify the LOSS function -> (ranking/mse)
            for epoch in range(self.num_of_epochs):
                # train the model
                self.model.train()

                for i, data in enumerate(meas_loader):
                    self.optimizer.zero_grad()
                    if self.loss_type == 'mse':
                        X_meas, y_meas = data
                        outputs = self.model(X_meas)
                        loss = self.loss_func(outputs.flatten(), y_meas.flatten())
                    elif self.loss_type == 'ranking':
                        x1, x2, y = data
                        y1 = self.model(x1)
                        y2 = self.model(x2)
                        loss = self.loss_func(y1.flatten(), y2.flatten(), y)

                    loss.backward()
                    self.optimizer.step()

            # just do greedy sampling (maximize the prediction)
            mu_avail = []
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(avail_loader):
                    X_avail, _ = data
                    X_avail = X_avail
                    y_avail = self.model(X_avail)  # (num_acq_samples, 1)
                    mu_avail.append(y_avail.detach().numpy())
            acq_vals = np.concatenate(mu_avail).flatten()

            # acq_vals = self.acq_func(
            #     mu_avail.flatten(), sigma_avail.flatten(), incumbent_scal
            # )  # (num_acq_samples,)

            # maximize for greedy
            # acq_vals = mu_avail.detach().numpy().flatten()
            # acq_vals = mu_avail
            # acq_vals = max(mu_avail.flatten())

            if self.goal == "maximize":
                # higher acq_vals the better
                sort_idxs = np.argsort(acq_vals)[::-1]  # descending order
                sample_idxs = sort_idxs[: self.batch_size]

            elif self.goal == "minimize":
                # lower acq_vals the better
                sort_idxs = np.argsort(acq_vals)  # ascending order
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

    def sample_meas_acq(self, avail_df, idx):
        """obtain the molecules suggested by the acquisition function"""
        return avail_df.iloc[idx, [0, 1, 2]]

    def sample_meas_randomly(self, avail_df):
        """take a single random sample from the available candiates"""
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx, [0, 1, 2]]

    def split_avail(self, df, observations):
        """return available and measured datasets"""
        data = df.data
        obs_smi = [o["smiles"] for o in observations]

        # avail_df is the set of molecules that have not been measured
        # create a function that checks if the smiles is in
        avail_df = data[~(data["smiles"].isin(obs_smi))]
        meas_df = data[data["smiles"].isin(obs_smi)]
        return meas_df, avail_df

    def make_xy(self, df, num=None):
        """generate featues and targets given a DataFrame"""
        y = df["target"].values.reshape(-1, 1)
        # if self.feature_type == "graphnet":
        #     # special treatment for GraphTuple features
        #     graphnet_list = df["feature"].tolist()
        #     if num is not None:
        #         graphnet_list = graphnet_list[: np.amin([num, len(graphnet_list)])]
        #         y = y[: np.amin([num, len(graphnet_list)])]
        #     else:
        #         pass
        #     X = utils_tf.concat(graphnet_list, axis=0)
        # else:
        #     # vector-valued features
        X = np.vstack(df["feature"].values)
        if num is not None:
            X = X[: np.amin([num, X.shape[0]]), :]
            y = y[: np.amin([num, X.shape[0]]), :]

        return X, y
