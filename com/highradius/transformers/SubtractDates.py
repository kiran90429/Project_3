import numpy as np
import datetime as dt
import pandas as pd


def SubtractDates(X, arguments=None):
     print(X[:,0])
     arguments=arguments.split(",")
     return np.array([(dt.datetime.strftime(pd.to_datetime(x),arguments[0]) - dt.datetime.strftime(pd.to_datetime(y),arguments[0])).days for x,y in zip(X[:,0],X[:,1])])

