import pandas as pd

def frac_top_n(dataset, bo_output, n, goal):
    df = pd.read_csv(dataset)
    #sort the dataset by target values in ascending order.
    if goal == 'maximize':
        df = df.nlargest(n, 'target', keep='first')
    if goal == 'minimize':
        df = df.nsmallest(n, 'target', keep='first')

    bo_output = pd.read_csv(bo_output)
    count = 0
    fracs = []
    for index, row in bo_output.iterrows():
        if row['target'] in df['target']:
            count += 1
        frac = count/n
        fracs.append(frac)
    bo_output['fracs'] = fracs
    
    return bo_output

def top_one(bo_output):
    df = pd.read_csv(bo_output)
    targets = df['target']
    cummax = targets.cummax()
    bo_output['cummax'] = cummax

    return bo_output


