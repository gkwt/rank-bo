import torch

# TODO -> old code with issues, need to be adapted
class BayesOptCampaign():
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
        dataset: torch.utils.dataset.Dataset,
        goal: str,
        model: torch.Models,
        acq_func_type,
        num_workers: int = 1,
        n_runs: int = 10,
        num_acq_samples: int = 50,
        batch_size: int = 1,
        budget: int = 100,
        init_design_strategy: str = 'random',
        num_init_design: int = 20,
        work_dir: str = '.',
        *args, 
        **kwargs,
    ):
        self.dataset = dataset
        self.goal = goal
        self.model_type = model_type
        self.feature_type = feature_type
        self.acq_func_type = acq_func_type
        self.num_acq_samples = num_acq_samples
        self.batch_size = batch_size
        self.budget = budget
        self.init_design_strategy = init_design_strategy
        self.work_dir = work_dir

        # process the dataset
        self._process_dataset()

        # check budget
        self._validate_budget()

        # initialize the surrogate model
        self._initialize_model(self.task_type)

        # register acquisition function
        self.acq_func = AcquisitionFunction(
            acq_func_type=self.acq_func_type, task_type=str(self.task_type), beta=self.beta
        )

    def _validate_budget(self):
        ''' validate that the budget does not exceed the total number of
        options in the dataset
        '''
        if self.budget > self.num_total:
            print(f'Requested budget exceeeds the number of total candidates. Resetting to {self.num_total}')
            self.budget = self.num_total - self.num_init_design

    def _process_dataset(self):
        ''' Process dataset object into pandas DataFrame used for 
        the sequential learning experiments. Generates dataframe containing the
        molecule smiles, feature representation and target values
        '''
        return


    def run(self, eval_metrics = False): 
        ''' Run the sequential learning experiments for num_restart independently seeded
        executions. The results for each run are stored in a list of pandas DataFrames and 
        saved in the pickle file res_file

        Args: 
            num_restarts (int): number of independent excecutions 
            eval_metrics (bool): toggles evaluating the surrogate on a held-out set (set to False
                if not looking at the performance/calibration, will save inference time)
        '''

        df_optimization = []            # collects optimization trace
        df_metrics = []                 # collects metrics during optimization (if specified)

        for num_restart in range(num_restarts):

            keep_running = True
            while keep_running:
                # try: 
                observations = []  # [ {'smiles': , 'y': }, ..., ]
                all_test_set_metrics = []

                iter_num = 0

                while (len(observations)-self.num_init_design) < self.budget:

                    # re-initialize the surrogate model
                    self._initialize_model(self.task_type)

                    meas_df, avail_df = self.split_avail(self.df, observations)
                    # shuffle the available candidates (acq func sampling)
                    avail_df = avail_df.sample(frac=1).reset_index(drop=True)

                    print(f'RESTART : {num_restart+1}\tNUM_ITER : {iter_num}\tNUM OBS : {len(observations)}')

                    if self.model_type == 'random':
                        # always use init design strategy
                        is_init_design = True
                    else:
                        is_init_design = len(observations) < self.num_init_design

                    if is_init_design:
                        # sample randomly
                        sample, measurement = self.sample_meas_randomly(avail_df)
                        observations.append({'smi': sample, 'y': measurement})
                    elif self.model_type == 'nn':
                        # use nearnest neighbour strategy
                        X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)
                        if self.goal == 'minimize':
                            best_fp = np.array(meas_df.nsmallest(n=1, columns='y')['x'].tolist()).squeeze()
                        elif self.goal == 'maximize':
                            best_fp = np.array(meas_df.nlargest(n=1, columns='y')['x'].tolist()).squeeze()
                        sims = np.array([chem.similarity_between_fps(x, best_fp) for x in X_avail])
                        ind = np.argsort(sims)[-self.batch_size:]
                        for sample_idx in ind:
                            sample, measurement = self.sample_meas_acq(avail_df, sample_idx)
                        observations.append({'smi': sample, 'y': measurement})
                    else:
                        # sample using surrogate model and acquisition
                        X_meas, y_meas = self.make_xy(meas_df)
                        X_avail, _ = self.make_xy(avail_df, num=self.num_acq_samples)

                        # get the scalers
                        X_scaler, y_scaler = get_scalers(
                            self.feature_type, self.model_type, self.task_type
                        )

                        # fit the scalers
                        X_scaler = X_scaler.fit(X_meas)
                        y_scaler = y_scaler.fit(y_meas)

                        # transform the current measurements
                        X_meas_scal = X_scaler.transform(X_meas)                    # (num_obs, X_dim)
                        y_meas_scal = y_scaler.transform(y_meas)                    # (num_obs, 1)

                        # transform the features of the available candidates
                        X_avail_scal = X_scaler.transform(X_avail)                  # (num_acq_samples, 1)

                        # get the scaled incumbent point
                        if self.goal == 'minimize':
                            incumbent_scal = np.amin(y_meas_scal)
                        elif self.goal == 'maximize':
                            incumbent_scal = np.amax(y_meas_scal)

                        # train the model on observations
                        self.model.train_bo(X_meas_scal, y_meas_scal)                   # (num_acq_samples, 1)

                        if self.task_type == enums.TaskType.regression:
                            mu_avail, sigma_avail = self.model.predict_bo(X_avail_scal)                         # (num_acq_samples, 1)
                            acq_vals = self.acq_func(mu_avail.flatten(), sigma_avail.flatten(), incumbent_scal) # (num_acq_samples,)
                        elif self.task_type == enums.TaskType.binary:
                            pred_avail, prob_avail = self.model.predict_bo(X_avail_scal)                         # (num_acq_samples, 1)
                            acq_vals = self.acq_func(prob_avail.flatten(), pred_avail.flatten(), incumbent_scal) # (num_acq_samples,)

                        if self.goal == 'minimize':
                            # higher acq_vals the better
                            sort_idxs = np.argsort(acq_vals)[::-1] # descending order
                            sample_idxs = sort_idxs[:self.batch_size]

                        elif self.goal == 'maximize':
                            # lower acq_vals the better
                            sort_idxs = np.argsort(acq_vals) # ascending order
                            sample_idxs = sort_idxs[:self.batch_size]

                        # perform measurements
                        for sample_idx in sample_idxs:
                            sample, measurement = self.sample_meas_acq(avail_df, sample_idx)
                            observations.append({'smi': sample, 'y': measurement})


                        # make a prediction on the heldout test set
                        if eval_metrics:
                            test_set_metrics = predict_test_set(
                                self.model, self.test_X, self.test_y, X_scaler, y_scaler, self.config,
                            )
                            test_set_metrics['iter_num'] = iter_num
                            all_test_set_metrics.append(test_set_metrics)

                    iter_num += 1

                # gather metrics if needed
                if eval_metrics and self.model_type not in ['random', 'nn']:
                    all_test_set_metrics = pd.concat(all_test_set_metrics)
                    all_test_set_metrics['run'] = num_restart
                    df_metrics.append(all_test_set_metrics)
                    
                data_dict = {}
                for key in observations[0].keys():
                    data_dict[key] = [o[key] for o in observations]
                data = pd.DataFrame(data_dict)
                if self.task_type == enums.TaskType.regression:
                    if self.goal == 'minimize':
                        data['trace'] = data['y'].cummin()
                    else:
                        data['trace'] = data['y'].cummax()
                elif self.task_type == enums.TaskType.binary:
                    if self.goal == 'minimize':
                        # find the negatives (ie. y==0)
                        data['trace'] = (data['y'] == 0).astype(int).cumsum()
                    else:
                        # find hte positives (ie. y==1)
                        data['trace'] = data['y'].cumsum()

                # statistics for the run
                data['model'] = self.model_type
                data['feature'] = self.feature_type
                data['dataset'] = self.dataset_name
                data['goal'] = self.goal
                data['acq_func'] = self.acq_func_type
                data['run'] = num_restart
                data['eval'] = range(0, len(data))
                df_optimization.append(data)
                
                keep_running = False

                # except:
                #     print('Run failed. Try again with different seed.')
        
        # gather all results as dataframe and return
        df_optimization = pd.concat(df_optimization)
        if df_metrics:
            df_metrics = pd.concat(df_metrics)

        return df_optimization, df_metrics



    def sample_meas_acq(self, avail_df, idx):
        ''' obtain the molecules suggested by the acquisition function 
        '''
        return avail_df.iloc[idx, [0, 2]]

    def sample_meas_randomly(self, avail_df):
        ''' take a single random sample from the available candiates
        '''
        idx = np.random.randint(avail_df.shape[0])
        return avail_df.iloc[idx, [0, 2]]

    def split_avail(self, df, observations):
        ''' return available and measured datasets 
        '''
        obs_smi = [o['smi'] for o in observations]

        avail_df = df[~(df['smi'].isin(obs_smi))]
        meas_df  = df[df['smi'].isin(obs_smi)]

        return meas_df, avail_df

    def make_xy(self, df, num=None):
        ''' generate featues and targets given a DataFrame
        '''
        y = df['y'].values.reshape(-1, 1)
        if self.feature_type == 'graphnet':
            # special treatment for GraphTuple features
            graphnet_list = df['x'].tolist()
            if num is not None:
                graphnet_list = graphnet_list[:np.amin([num, len(graphnet_list)])]
                y = y[:np.amin([num, len(graphnet_list)])]
            else:
                pass
            X = utils_tf.concat(graphnet_list, axis=0)
        else:
            # vector-valued features
            X = np.vstack(df['x'].values)
            if num is not None:
                X = X[:np.amin([num, X.shape[0]]), :]
                y = y[:np.amin([num, X.shape[0]]), :]

        return X, y


