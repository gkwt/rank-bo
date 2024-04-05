#/bin/bash

# for running on mariana CS cluster
# module load anaconda
source /opt/intel/oneapi/setvars.sh
source /opt/python/3.8a/bin/activate

# create and activate
conda create -n rbbo python=3.8
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/$USER/.conda/envs/rbbo/lib
conda activate rbbo

# install conda packages first
conda install -c conda-forge scikit-learn pandas seaborn matplotlib
conda install -c conda-forge rdkit        # was not working for some reason
conda install -c pytorch pytorch

# install pip packages next
# pip install rdkit-pypi      # if the conda version is not working
pip install mordred
pip install rogi
pip install PyTDC
pip install bayesian-torch
# pip install pytorchltr
# pip install chemprop

# deactivate
conda deactivate 