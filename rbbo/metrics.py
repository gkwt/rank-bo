import pandas as pd
from sklearn.metrics import auc 
from sklearn.preprocessing import minmax_scale

def frac_top_n(
    df: pd.DataFrame, 
    bo_output: pd.DataFrame, 
    n: int, 
    goal: str = 'maximize'
):
    if goal == "maximize":
        df = df.nlargest(n, "target", keep="first")
    elif goal == "minimize":
        df = df.nsmallest(n, "target", keep="first")

    count = 0
    fracs = []
    for index, row in bo_output.iterrows():
        if row["smiles"] in df["smiles"].tolist():
            count += 1
        frac = count / float(n)
        fracs.append(frac)
    bo_output["frac_top_n"] = fracs

    return bo_output


def top_one(
    bo_output: pd.DataFrame, 
    goal: str = 'maximize'
):
    targets = bo_output["target"]
    if goal == 'maximize':
        bo_output["top_one"] = targets.cummax()
    elif goal == 'minimize':
        bo_output["top_one"] = targets.cummin()
    return bo_output

def frac_top_n_percent(
        df: pd.DataFrame, 
        bo_output: pd.DataFrame, 
        n: int, 
        goal: str = 'maximize'
):

    if goal == 'maximize':
        q = 1 - (n/100)
        quantile = df['target'].quantile(q)
        df = df[df["target"] > quantile]
    elif goal == 'minimize':
        q = n/100
        quantile = df['target'].quantile(q)
        df = df[df["target"] < quantile]

    count = 0
    fracs = []
    length = len(df)
    for index, row in bo_output.iterrows():
        if row["smiles"] in df["smiles"].tolist():
            count += 1
        frac = count / float(length)
        fracs.append(frac)
    bo_output["frac_top_n_percent"] = fracs

    return bo_output

def custom_auc(
        bo_output: pd.DataFrame, 
        metric: str = "frac_top_n"
               
):
    bo_output = bo_output.reset_index()
    x = bo_output['index']
    y = bo_output[metric]

    x_scaled = minmax_scale(x)
    y_scaled = minmax_scale(y)

    custom_auc = auc(x_scaled, y_scaled)

    return custom_auc