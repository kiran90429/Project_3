# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:27:10 2017

@author: BharatChandra.T
"""
from com.highradius.Metrics.ConfusionMatrixRowSumMetric import ConfusionMatrixRowSumMetric
from com.highradius.Metrics.ConfusionMatrixColSumMetric import ConfusionMatrixColSumMetric
from com.highradius.Metrics.ConfusionMatrixPredictedCorrectMetric import ConfusionMatrixPredictedCorrectMetric
from com.highradius.Reports.BaseReport import BaseReport


class StandardClassificationReport(BaseReport):
    """
    * This class is an extended class of BaseReport class which will generate the CSV file specific to Classification problem.
    """
    def __init__(self, filepath, classnames):
        """
        :param filepath: Should pass the filepath to save the csv report.
        :param classnames: Should pass class names
        :return: This will super BaseReport _init_ method with filepath,list of metrics to evaluate and generate report.
        """
        self.classnames = classnames
        self.metrics=[]
        columnbers = len(self.classnames)
        for num in range(columnbers):
            self.metrics.append([self.classnames[num] + ' ' + ConfusionMatrixRowSumMetric.name, ConfusionMatrixRowSumMetric(num)])
            self.metrics.append([self.classnames[num] + ' ' + ConfusionMatrixColSumMetric.name, ConfusionMatrixColSumMetric(num)])
            self.metrics.append([self.classnames[num] + ' ' + ConfusionMatrixPredictedCorrectMetric.name, ConfusionMatrixPredictedCorrectMetric(num)])
        super().__init__(filepath,self.metrics)

