#/bin/bash

# for running on mariana CS cluster
source /opt/python/3.8a/bin/activate
# module load cuda/11.1

# create and activate
conda create -n rbbo python=3.8 -y
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/$USER/.conda/envs/rbbo/lib
conda activate rbbo

# besure to install pytorch with correct cuda version before pytorch-scatter
conda install -c pytorch -c nvidia pytorch-cuda=12.1 -y
conda install -c pyg pyg=*=*cu* -y
# conda install -c pytorch pytorch cudatoolkit=11.3
# conda install -c pyg pytorch-scatter -y
conda install -c gpytorch gpytorch -y

# install conda packages first
conda install -c conda-forge scikit-learn pandas seaborn matplotlib scipy -y
conda install -c conda-forge rdkit=2024.03.1 -y

# install pip packages next
# pip install rogi
# pip install PyTDC
pip install gauche bayesian-torch --no-deps     # suppress dependencies to avoid 
                                                # affecting conda installs of pytorch

# other packages
# pip install rdkit-pypi      # if the conda version is not working
# pip install mordred
# pip install bayesian-torch
# pip install gpytorch
# pip install pytorchltr
# pip install git+https://github.com/bp-kelley/descriptastorus
# pip install chemprop

# deactivate
conda deactivate 