import pandas as pd


def frac_top_n(
    df: pd.DataFrame, 
    bo_output: pd.DataFrame, 
    n: int, 
    goal: str = 'maximize'
):
    # df = pd.read_csv(dataset)
    # sort the dataset by target values in ascending order.
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
    bo_output["fracs"] = fracs

    return bo_output


def top_one(
    bo_output: pd.DataFrame, 
    goal: str = 'maximize'
):
    df = bo_output
    targets = df["target"]
    if goal == 'maximize':
        bo_output["top1"] = targets.cummax()
    elif goal == 'minimize':
        bo_output["top1"] = targets.cummin()
    return bo_output

def frac_top_n_percent(dataset, bo_output, n, goal):
    df = pd.read_csv(dataset)
    if goal == 'maximize':
        q = 1 - (n/100)
        quantile = df['target'].quantile(q)
        df = df[df["target"] > quantile]
    elif goal == 'minimize':
        q = n/100
        quantile = df['target'].quantile(q)
        df = df[df["target"] < quantile]

    bo_output = pd.read_csv(bo_output)
    count = 0
    fracs = []
    length = len(df)
    for index, row in bo_output.iterrows():
        if row["smiles"] in df["smiles"].tolist():
            count += 1
        frac = count / float(length)
        fracs.append(frac)
    bo_output["fracs"] = fracs

    return bo_output