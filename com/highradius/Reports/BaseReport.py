# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:24:22 2017

@author: bharatchandra.t
@Description : This is a BaseReport class which will create or append data to the data farme to create CSV file as a report.
"""
import pandas as pd


class BaseReport:
    """
    * This is a BaseReport class which will evaluate metrics specific to the problem and generates the report in CSV format.
    """
    def __init__(self,filepath,metrics):
        """
        :param df: This is a local variable which create an empty dataframe
        :param filepath: This will be super from problem specific report file
        :param metrics: This will be super from problem specific report file
        """
        self.df = pd.DataFrame()
        self.metrics =metrics
        self.indexnumber = 0
        self.filepath = filepath

    def create_csv(self):
        """
        :return: This will save the CSV file in the specified location and also prints the success message with filepath.
        """
        with open(self.filepath, "w") as f:
            self.df.to_csv(f, index=False,header=True,columns=self.columns, encoding='utf-8')
        print ("Generated the report in the following location %s" %self.filepath)
        
    def appendtodataframe(self, y_true, y_predicted, testcase):
        """
        :param y_true: True labels
        :param y_predicted: True predicted values
        :param testcase: gets the testcase
        """
        self.columns =['TestCases']
        self.values =[]
        self.values.append(testcase)
        for name,metric in self.metrics:
            self.columns.append(name)
            self.values.append(metric.evaluate(y_true, y_predicted))
        dictionary = dict(zip(self.columns, self.values))
        df1 = pd.DataFrame(dictionary, index=[self.indexnumber])
        self.df = self.df.append(df1)
        self.indexnumber += 1


