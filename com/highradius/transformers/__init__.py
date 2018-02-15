from sklearn.base import TransformerMixin

import numpy as np
import pandas as pd
import datetime as dt
import os


class CustomTransformFunctionGenerator(TransformerMixin):
    def __init__(self, function, arguments=None):
        self.function = function
        self.arguments = arguments

    def get_attribute(self, kls):
        parts = kls.split('.')
        module = ".".join(parts[:-1])
        print(os.getcwd())
        m = __import__(module)

        for comp in parts[1:]:
            m = getattr(m, comp)
        return m

    def transform(self, X, *_):
        print(self.function + "." + self.function)
        if self.function == '-':
            transformerFunction = self.get_attribute(
                "com.highradius.transformers." + "LitmRowNumber" + "." + "LitmRowNumber")
        elif self.function == '/':
            transformerFunction = self.get_attribute(
                "com.highradius.transformers." + "DivisionByInvoiceAmount" + "." + "DivisionByInvoiceAmount")
        else:
            transformerFunction = self.get_attribute(
                "com.highradius.transformers." + self.function + "." + self.function)
        print(transformerFunction(X, self.arguments))
        return transformerFunction(X, self.arguments)

    def fit(self, *_):
        return self