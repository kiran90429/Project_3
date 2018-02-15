import numpy as np
import datetime as dt
import pandas as pd

def SubtractDatesPG(X,arguments=None):

     date1 = pd.to_datetime(X[:,0].astype('str')).strftime('%Y%m%d')
     date2 = pd.to_datetime(X[:,1].astype('str')).strftime('%Y%m%d')

     #
     # sub_check = pd.DataFrame()
     #
     # sub_check['date1'] = pd.Series(date1)
     # sub_check['date2'] = pd.Series(date2)
     #

     return np.array([(x - y).days for x,y in zip(pd.to_datetime(date1,format="%Y%m%d"),pd.to_datetime(date2,format="%Y%m%d"))])

