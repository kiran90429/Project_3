import numpy as np
import ast
import pandas as pd

def MonthOfYear(X,arguments=None):
    m=pd.Series(X.tolist())
    print(m)

    m = pd.to_datetime(m,format="%Y-%m-%d")
    return m.dt.month

