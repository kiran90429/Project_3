import numpy as np
import ast
import pandas as pd

def HistoryKunwe(X,arguments=None):
    arguments= "{" + arguments + "}"
    history_mapping=ast.literal_eval(arguments)
    m=pd.Series(X.tolist()).astype('str').str.split('.',0).str[0]
    #m=m.apply(lambda a: history_mapping[a] if a in history_mapping else np.mean(list(history_mapping.values())))
    m = m.apply(lambda a: history_mapping[a] if a in history_mapping else history_mapping['others'])
    # print(m)

    u = pd.DataFrame(columns=['kunwe','history'])

    u['kunwe'] = pd.Series(X.tolist())
    u['history'] = m
    u.to_csv('history_kunwe.csv')
    print(np.unique(m))
    return m

