import numpy as np
import datetime as dt


def DayOfWeek(X,arguments=None):
     y = []
     for i in range(len(X)):
         x  = dt.datetime.strptime(str(X[i]),"%Y%m%d").weekday()
         y.append(x)
     y = np.array(y)
     print(y)
     return y







