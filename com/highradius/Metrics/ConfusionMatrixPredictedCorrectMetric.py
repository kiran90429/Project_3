# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:06:32 2017

@author: BharatChandra.T
"""
from sklearn.metrics import confusion_matrix


class ConfusionMatrixPredictedCorrectMetric:
    """
        * This is a metric class which will return the Predicted Correct (diagonal value) of each class in confusion matrix.
        * This class have one variable,method.
    """

    def __init__(self,columnnumber):
        self.columnnumber = columnnumber

    name = 'Predicted Correct'

    def evaluate(self,y_true,y_predicted):
        """
        :param y_true: True labels
        :param y_predicted: True predicted values
        :param i: class number (depends on number classes confusion matrix have)
        :return: predicted correct(diagonal value) of each class
        """
        confmatirx = confusion_matrix(y_true, y_predicted)
        self.confmatrixlist = (list(confmatirx))
        return self.confmatrixlist[self.columnnumber][self.columnnumber]