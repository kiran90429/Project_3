# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:27:10 2017

@author: BharatChandra.T
"""
from com.highradius.Metrics.RmseMetric import RmseMetric
from com.highradius.Metrics.ErrorDays import ErrorDays
from com.highradius.Reports.BaseReport import BaseReport


class StandardCollectionsReport(BaseReport):
    """
    * This class is an extended class of BaseReport class which will generate the CSV file specific to Collections problem.
    """
    def __init__(self,filepath,days):
        """
        :param filepath: Should pass the filepath to save the csv report.
        :param days: Should pass specific days
        :return: This will super BaseReport _init_ method with filepath,list of metrics to evaluate and generate report.
        """
        self.days = days
        self.metrics =[]
        self.metrics.append([RmseMetric.name, RmseMetric()])
        for day in days:
            self.metrics.append(['+/-' + ' '+str(day)+ErrorDays.name,ErrorDays(day)])
        super().__init__(filepath, self.metrics)