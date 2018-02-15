# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:06:32 2017

@author: BharatChandra.T
"""
from sklearn.metrics import confusion_matrix


class ConfusionMatrixColSumMetric:
    """
    * This is a metric class which will return the column sum of each class in confusion
    * This class have one variable contains name of the metrics,a method to evaluate the metric
    """

    def __init__(self,columnnumber):
        self.columnnumber = columnnumber

    name = 'Column Sum'

    def evaluate(self,y_true,y_predicted):
        """
        :param y_true: True labels
        :param y_predicted: True predicted values
        :param i: class number (depends on number classes confusion matrix have)
        :return: column sum of each class
        """
        confmatirx = confusion_matrix(y_true, y_predicted)
        self.confmatrixlist = (list(confmatirx))
        columscore = 0
        self.totalclasses = confmatirx.shape[1]
        for j in range(self.totalclasses):
            columscore += (self.confmatrixlist[j][self.columnnumber])
        return columscore