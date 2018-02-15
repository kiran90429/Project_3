# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:06:32 2017

@author: BharatChandra.t
"""
from sklearn.metrics import confusion_matrix


class ConfusionMatrixRowSumMetric:
    """
    * This is a metric class which will return the column sum of each class in confusion
    * This class have one variable,method
    """
    def __init__(self,columnnumber):
        self.columnnumber = columnnumber

    name = 'Row Sum'

    def evaluate(self,y_true,y_predicted):
        """
        :param y_true: True labels
        :param y_predicted: True predicted values
        :param i: class number (depends on number classes confusion matrix have)
        :return: row sum of each class
        """
        confmatirx = confusion_matrix(y_true, y_predicted)
        self.metricconf = (list(confmatirx))
        return self.metricconf[self.columnnumber].sum()

