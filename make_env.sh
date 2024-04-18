#/bin/bash

# for running on mariana CS cluster
source /opt/python/3.8a/bin/activate
module load cuda/11.3

# create and activate
conda create -n rbbo python=3.8 -y
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/$USER/.conda/envs/rbbo/lib
conda activate rbbo

# install conda packages first
conda install -c conda-forge scikit-learn pandas seaborn matplotlib scipy -y
conda install -c conda-forge rdkit -y
conda install -c pyg pyg -y

# besure to install pytorch with correct cuda version before pytorch-scatter
conda install -y pytorch=*=*cuda* cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-scatter -y
conda install -c gpytorch gpytorch -y
# conda install -c conda-forge blitz-bayesian-pytorch -y

# install pip packages next
pip install rogi
pip install gauche bayesian-torch --no-deps     # suppress dependencies to avoid 
                                                # affecting conda installs of pytorch

# other packages
# pip install rdkit-pypi      # if the conda version is not working
# pip install mordred
# pip install PyTDC
# pip install bayesian-torch
# pip install gpytorch
# pip install pytorchltr
# pip install git+https://github.com/bp-kelley/descriptastorus
# pip install chemprop

# deactivate
conda deactivate 