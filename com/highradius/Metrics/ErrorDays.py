# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:39:54 2017
@author: BharatChandra.T
"""
import math


class ErrorDays:
    """
    * This is a metric class which will return the days accuracy.
    * This class have one variable which contains name of the metric and a method to evaluate the metric.
    """
    def __init__(self,accuracyday):
        self.accuracyday =accuracyday

    name = 'Days'

    def evaluate(self,y_true,y_predicted):
        """
        :param y_true: True labels
        :param y_predicted: True predicted values
        :param value: Should pass number of days
        :return: Accuracy of the value passed
        """
        total = len(y_true)
        count = 0
        for y_test, y_pred in zip(y_true, y_predicted):
            if ((y_test - y_pred) ** 2) < math.pow(self.accuracyday, 2):
                count += 1
        return count / total
