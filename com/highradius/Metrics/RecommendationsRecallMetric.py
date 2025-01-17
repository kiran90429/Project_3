# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:06:32 2017

@author: BharatChandra.T
"""


class RecommendationsRecallMetric:
    """
    * This is a metric class which will return the column sum of each class in confusion
    * This class have one variable contains name of the metrics,a method to evaluate the metric
    """
    name = 'Recall'

    def evaluate(self,resolutions,suggestions):
        """
        :param resolutions: Need to pass resolutions
        :param suggestions: Need to pass suggestions
        :return: return's Recall value
        """
        x = set(resolutions) & set(suggestions)
        recall = len(x) / len(resolutions)
        return recall
