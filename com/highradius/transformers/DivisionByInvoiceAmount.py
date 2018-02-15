import numpy as np
import ast
import pandas as pd

def DivisionByInvoiceAmount(X,arguments=None):

    m1=pd.Series(X[:,0].tolist())
    m2 = pd.Series(X[:,1].tolist())
    m = m1/m2
    return m

