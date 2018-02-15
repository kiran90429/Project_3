import numpy as np
import datetime as dt


def DayOfMonth(X,arguments=None):
     y = []
     for i in range(len(X)):
         x  = dt.datetime.strptime(str(X[i]),"%Y%m%d").day
         y.append(x)
     y = np.array(y)
     return y
     