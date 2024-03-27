import torch
import numpy as np
import pandas as pd


# TODO -> old code with issues, need to be adapted
class BayesOptCampaign:
    """
    Args:
        dataset_name (str): the name of the DIONYSUS dataset
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
        model: torch.nn.Module,
        feature_type: str,
        loss_func: str,
        acq_func_type,
        num_workers: int = 1,
        num_of_epochs: int = 5,
        num_total: int = 10,
        n_runs: int = 10,
        num_acq_samples: int = 50,
        batch_size: int = 1,
        budget: int = 100,
        init_design_strategy: str = "random",
        num_init_design: int = 20,
        work_dir: str = ".",
        *args,
        **kwargs,
    ):
        self.dataset = dataset
        self.goal = goal
        self.model = model
        self.feature_type = feature_type
        self.acq_func_type = acq_func_type
        self.num_acq_samples = num_acq_samples
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir
        self.num_init_design = num_init_design
        self.loss_func = loss_func
        self.num_workers = num_workers
        self.num_of_epochs = num_of_epochs
        self.n_runs = n_runs
        self.num_total = num_total

        # process the dataset
        self._process_dataset()

        # check budget
        # self._validate_budget()

        # initialize the surrogate model
        # self._initialize_model(self.task_type)

        # register acquisition function
        # self.acq_func = AcquisitionFunction(
        #     acq_func_type=self.acq_func_type,
        #     task_type=str(self.task_type),
        #     beta=self.beta,
        # )

    def _validate_budget(self):
        """validate that the budget does not exceed the total number of
        options in the dataset
        """
        if self.budget > self.num_total:
            print(
                f"Requested budget exceeeds the number of total candidates. Resetting to {self.num_total}"
            )
            self.budget = self.num_total - self.num_init_design

    def _process_dataset(self):
        """Process dataset object into pandas DataFrame used for
        the sequential learning experiments. Generates dataframe containing the
        molecule smiles, feature representation and target values
        """
        return

    def _reinitialize_model(self):
        """Reinitialize the model from scratch (re-train)"""
        reinit_model = self.model
        for layer in reinit_model.layers:
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        return reinit_model

    def run(self, num_restarts: int, eval_metrics=False):
        """Run the sequential learning experiments for num_restart independently seeded
        executions. The results for each run are stored in a list of pandas DataFrames and
        saved in the pickle file res_file

        Args:
            num_restarts (int): number of independent excecutions
            eval_metrics (bool): toggles evaluating the surrogate on a held-out set (set to False
                if not looking at the performance/calibration, will save inference time)
        """

        df_optimization = []  # collects optimization trace
        df_metrics = []  # collects metrics during optimization (if specified)

        for num_restart in range(num_restarts):

            observations = []  # [ {'smiles': , 'y': }, ..., ]
            all_test_set_metrics = []
            iter_num = 0

            while (len(observations) - self.num_init_design) < self.budget:

                # re-initialize the surrogate model
                if iter_num >= 1:
                    self.model = self._reinitialize_model()
                # split dataset into measured and available candidates
                meas_df, avail_df = self.split_avail(self.dataset, observations)
                # shuffle the available candidates (acq func sampling)
                avail_df = avail_df.sample(frac=1).reset_index(drop=True)
                print(
                    f"RESTART : {num_restart+1}\tNUM_ITER : {iter_num}\tNUM OBS : {len(observations)}"
                )

                # sample randomly
                sample, measurement = self.sample_meas_randomly(avail_df)
                observations.append({"smi": sample, "y": measurement})
                # elif self.model_type == "nn":
                #     # use nearnest neighbour strategy
                #     X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)
                #     if self.goal == "minimize":
                #         best_fp = np.array(
                #             meas_df.nsmallest(n=1, columns="y")["x"].tolist()
                #         ).squeeze()
                #     elif self.goal == "maximize":
                #         best_fp = np.array(
                #             meas_df.nlargest(n=1, columns="y")["x"].tolist()
                #         ).squeeze()
                #     sims = np.array(
                #         [chem.similarity_between_fps(x, best_fp) for x in X_avail]
                #     )
                #     ind = np.argsort(sims)[-self.batch_size :]
                #     for sample_idx in ind:
                #         sample, measurement = self.sample_meas_acq(
                #             avail_df, sample_idx
                #         )
                #     observations.append({"smi": sample, "y": measurement})

                # sample using surrogate model and acquisition
                X_meas, y_meas = self.make_xy(meas_df)
                X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)

                # SCALING NOT REQUIRED
                # get the scalers
                # X_scaler, y_scaler = get_scalers(
                #     self.feature_type, self.model_type, self.task_type
                # )

                # # fit the scalers
                # X_scaler = X_scaler.fit(X_meas)
                # y_scaler = y_scaler.fit(y_meas)

                # # transform the current measurements
                # X_meas_scal = X_scaler.transform(X_meas)  # (num_obs, X_dim)
                # y_meas_scal = y_scaler.transform(y_meas)  # (num_obs, 1)

                # # transform the features of the available candidates
                # X_avail_scal = X_scaler.transform(
                #     X_avail
                # )  # (num_acq_samples, 1)

                # get the scaled incumbent point
                # if self.goal == "minimize":
                #     incumbent_scal = np.amin(y_meas_scal)
                # elif self.goal == "maximize":
                #     incumbent_scal = np.amax(y_meas_scal)

                # train the model on observations
                # TODO training loop required here -> 5 epochs
                # start with fresh model every time
                # specify the LOSS function -> (ranking/mse)
                for epoch in range(self.num_of_epochs):
                    # train the model
                    self.model.train(True)
                    self.optimizer.zero_grad()
                    outputs = self.model(X_meas)
                    loss = self.loss_func(outputs, y_meas)
                    loss.backward()
                    self.optimizer.step()

                # TODO Inference
                # just do greedy sampling (maximize the prediction)
                self.model.train(False)
                mu_avail = self.model(X_avail)  # (num_acq_samples, 1)
                # acq_vals = self.acq_func(
                #     mu_avail.flatten(), sigma_avail.flatten(), incumbent_scal
                # )  # (num_acq_samples,)

                # maximize for greedy
                acq_vals = max(mu_avail.flatten())

                if self.goal == "minimize":
                    # higher acq_vals the better
                    sort_idxs = np.argsort(acq_vals)[::-1]  # descending order
                    sample_idxs = sort_idxs[: self.batch_size]

                elif self.goal == "maximize":
                    # lower acq_vals the better
                    sort_idxs = np.argsort(acq_vals)  # ascending order
                    sample_idxs = sort_idxs[: self.batch_size]

                # TODO make "measurement"
                # perform measurements
                for sample_idx in sample_idxs:
                    sample, measurement = self.sample_meas_acq(avail_df, sample_idx)
                    observations.append({"smi": sample, "y": measurement})

                df_optimization.append(pd.DataFrame(observations))

    def sample_meas_acq(self, avail_df, idx):
        """obtain the molecules suggested by the acquisition function"""
        return avail_df.iloc[idx, [0, 2]]

    def sample_meas_randomly(self, avail_df):
        """take a single random sample from the available candiates"""
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx, [0, 2]]

    def split_avail(self, df, observations):
        """return available and measured datasets"""
        data = df.data
        obs_smi = [o["smiles"] for o in observations]

        avail_df = data[~(data["smiles"].isin(obs_smi))]
        meas_df = data[data["smiles"].isin(obs_smi)]
        print(meas_df, avail_df)
        return meas_df, avail_df

    def make_xy(self, df, num=None):
        """generate featues and targets given a DataFrame"""
        y = df["y"].values.reshape(-1, 1)
        # if self.feature_type == "graphnet":
        #     # special treatment for GraphTuple features
        #     graphnet_list = df["x"].tolist()
        #     if num is not None:
        #         graphnet_list = graphnet_list[: np.amin([num, len(graphnet_list)])]
        #         y = y[: np.amin([num, len(graphnet_list)])]
        #     else:
        #         pass
        #     X = utils_tf.concat(graphnet_list, axis=0)
        # else:
        #     # vector-valued features
        X = np.vstack(df["x"].values)
        if num is not None:
            X = X[: np.amin([num, X.shape[0]]), :]
            y = y[: np.amin([num, X.shape[0]]), :]

        return X, y
