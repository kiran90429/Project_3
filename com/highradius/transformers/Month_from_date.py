import numpy as np
import datetime as dt
import pandas as pd


def Month_from_date(X,arguments=None):
    m = pd.Series(X.tolist()).dt.month
    return m




