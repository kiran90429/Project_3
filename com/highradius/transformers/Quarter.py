import numpy as np
import datetime as dt


def Quarter(X,arguments=None):
    y = []
    for i in range(len(X)):
        x  = ((dt.datetime.strptime(str(X[i]),"%Y%m%d").month-1)//3)+1
        y.append(x)
    y = np.array(y)
    return y




