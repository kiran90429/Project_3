import numpy as np
import ast
import pandas as pd

def HistoryXref(X,arguments=None):
    arguments= "{" + arguments + "}"
    history_mapping=ast.literal_eval(arguments)
    m=pd.Series(X.tolist())
    #m=m.apply(lambda a: history_mapping[a] if a in history_mapping else np.mean(list(history_mapping.values())))
    m = m.apply(lambda a: history_mapping[a] if a in history_mapping else history_mapping['others'])
    print(m)

    u = pd.DataFrame(columns=['xref','history'])

    u['xref'] = pd.Series(X.tolist())
    u['history'] = m
    u.to_csv('history_xref.csv')
    return m

