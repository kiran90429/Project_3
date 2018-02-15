import numpy as np
import ast
import pandas as pd

def History(X,arguments=None):
    arguments= "{" + arguments + "}"
    history_mapping=ast.literal_eval(arguments)
    m=pd.Series(X.tolist())
    m = m.astype('int')
    #m=m.apply(lambda a: history_mapping[a] if a in history_mapping else np.mean(list(history_mapping.values())))
    m = m.apply(lambda a: history_mapping[a] if a in history_mapping else history_mapping[-1])
    print(m)
    i = pd.DataFrame(columns=['KUNWE','values'])
    i['KUNWE'] = pd.Series(X.tolist()).astype(int)
    i['values'] = m

    i.to_csv('pmml_history.csv')
    return np.round(m,4)

