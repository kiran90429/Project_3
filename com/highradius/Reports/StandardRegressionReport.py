# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:27:10 2017

@author: BharatChandra.T
"""
from com.highradius.Metrics.RmseMetric import RmseMetric
from com.highradius.Reports.BaseReport import BaseReport


class StandardRegressionReport(BaseReport):
    """
    * This class is an extended class of BaseReport class which will generate the CSV file specific to Regression problem.
    """
    def __init__(self,filepath):
        """
        :param filepath: Should pass filepath to save the csv report.
        :return: This will super BaseReport _init_ method with filepath,list of metrics to evaluate and generate report.
        """
        metricobjects = [RmseMetric()]
        self.metrics =[]
        for metric in metricobjects:
            name = metric.name
            self.metrics.append([name,metric])
        super().__init__(filepath, self.metrics)
