# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:39:54 2017
@author: BharatChandra.T
"""
import math
from sklearn.metrics import mean_squared_error


class RmseMetric:
    """
    * This is a metric class which will return the Root Mean Square Error.
    * This class have one variable which contains name of the metric and a method to evaluate the metric.
    """
    name = 'RMSE'

    def evaluate(self,y_true,y_predicted):
        """
        :param y_true: True labels
        :param y_predicted: Predicted true values
        :return: Returns the RMSE in 2 decimal value
        """
        value = math.sqrt(mean_squared_error(y_true, y_predicted))
        return round(value, 2)

