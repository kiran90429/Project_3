import numpy as np
import ast
import pandas as pd

def HistoryAmount(X,arguments=None):
    arguments= "{" + arguments + "}"
    history_mapping=ast.literal_eval(arguments)

    kunnr=pd.Series(X[:,0].tolist(),name='kunnr')

    kunnr = kunnr.astype(int)
    kunnr = kunnr.apply(lambda a: a if a in history_mapping.keys() else 999999999)
    mean_amount=kunnr.apply(lambda a: history_mapping[a] if a in history_mapping else history_mapping[999999999])

    amount = pd.Series(X[:,1].tolist(),name='amount')
    #x = pd.concat([kunnr,mean_amount],axis=1)

    #x['amount'] = x[:,1].tolist()

    #x = pd.concat([kunnr,mean_amount,amount],axis=1)

    print(amount/mean_amount)
    k = pd.DataFrame(columns=['amount/mean_amount'],data=amount/mean_amount)
    k.to_csv('original_dispute_amount.csv')
    return np.round(amount/mean_amount,4)

