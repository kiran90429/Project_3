#pmml code :
import pandas as pd
from sklearn2pmml import sklearn2pmml

from sklearn2pmml import DataFrameMapper

from sklearn2pmml import PMMLPipeline

from sklearn2pmml.decoration import CategoricalDomain
from sklearn2pmml.decoration import ContinuousDomain

from sklearn.ensemble import RandomForestClassifier

from com.highradius.transformers import CustomTransformFunctionGenerator

train = pd.read_csv('data/training_set_for_pmml.csv',encoding ='latin')


train['main_output_1']=(train['FIN_PAID_AMT']>(0.01*train['FIN_ORIGINAL_AMT']))
train['main_output_1']=train['main_output_1'].map({True:-1,False:1})


train = train.rename(columns={'ZZ_CLAIMDATE_SIMP_DT': 'customer_claim_date', 'CREATE_TIME': 'deduction_created_date','ZZ_XREF3': 'product_category','KUNWE': 'ship_to','FIN_ORIGINAL_AMT': 'original_dispute_amount','FIN_KUNNR': 'payer','FIN_PAID_AMT': 'paid_amount','main_output_1':'labels'})


#
kunwe_top = pd.read_csv('history2/ship_to_history.csv')

kunwe_top = kunwe_top[['KUNWE','ship_to_history']]

kunwe_top = kunwe_top.sort_values(by='KUNWE',ascending=False)

kunwe_top['KUNWE'] = kunwe_top['KUNWE'].astype('str')
# #
import numpy as np

kunwe_dict = dict([(kunwe,history)for kunwe,history in zip(kunwe_top['KUNWE'],kunwe_top['ship_to_history'])])

kunwe_dict = str(kunwe_dict).replace("{","").replace("}","")

xref_history = pd.read_csv('history2/category_history.csv')

xref_history = xref_history[['ZZ_XREF3','category_history']]

xref_history.sort_values(by='ZZ_XREF3',ascending=False,inplace=True)

#xref_history['category_history'] = np.round(xref_history['category_history'],5)

xref_dict = dict([(xref,history)for xref,history in zip(xref_history['ZZ_XREF3'],xref_history['category_history'])])

xref_dict = str(xref_dict).replace("{","").replace("}","")
# #
mean_amount_history = pd.read_csv('history2/dispute_history_mean_table.csv')


mean_amount_history = mean_amount_history[['FIN_KUNNR_hist_group','dispute_mean']]

mean_amount_history = mean_amount_history.sort_values(by='FIN_KUNNR_hist_group')

#mean_amount_history['dispute_mean'] = np.round(mean_amount_history['dispute_mean'],5)

mean_amount_dict = dict([(kunnr,mean_amount) for kunnr,mean_amount in zip(mean_amount_history['FIN_KUNNR_hist_group'],
                                                                          mean_amount_history['dispute_mean'])])
mean_amount_dict = str(mean_amount_dict).replace("{","").replace("}","")
# #
category_rank = pd.read_csv('history2/category_rank_table.csv')

category_rank['combined'] = category_rank['FIN_KUNNR_hist_group'].astype('str') + '|' + category_rank['ZZ_XREF3'] +'|'+ category_rank['xref_label'].astype('str')

category_rank = category_rank[['combined','rank_xref_in_kunnr']]

category_rank_dict = dict([(combined,rank) for combined,rank in zip(category_rank['combined'],category_rank['rank_xref_in_kunnr'])])

category_rank_dict = str(category_rank_dict).replace("{","").replace("}","")
#
kunwe_rank = pd.read_csv('history2/rank_in_kunwe.csv')

kunwe_rank['combined'] = kunwe_rank['FIN_KUNNR_hist_group'].astype('str') + '|' + kunwe_rank['KUNWE'].astype('str')+'|'+ kunwe_rank['kunwe_label'].astype('str')

kunwe_rank_dict = dict([(combined,rank) for combined,rank in zip(kunwe_rank['combined'],kunwe_rank['rank_kunwe_in_kunnr'])])

kunwe_rank_dict = str(kunwe_rank_dict).replace("{","").replace("}","")
#
cal_customer_history = pd.read_csv('history2/cal_customer_history.csv')

cal_customer_history['combined'] = cal_customer_history['FIN_KUNNR'] + '|' + cal_customer_history['KUNWE'] + '|' + cal_customer_history['ZZ_XREF3']

cal_customer_history = cal_customer_history[['combined','cal_cust_history']]

#cal_customer_history['cal_cust_history'] = np.round(cal_customer_history['cal_cust_history'],5)

cal_customer_history_dict = dict([(combined,cal_customer_history) for combined,cal_customer_history in zip(cal_customer_history['combined'],cal_customer_history['cal_cust_history'])])

cal_customer_history_dict = str(cal_customer_history_dict).replace("{","").replace("}","")
#

#b_value
b_value = pd.read_csv('history2/b_value_history.csv')

b_value['combined'] = b_value['FIN_KUNNR_hist_group'].astype('str') + '|' + b_value['value_label']

b_value = b_value[['combined','avg_invalid_dispute_amount']]

#b_value['avg_invalid_dispute_amount'] = np.round(b_value['avg_invalid_dispute_amount'],5)

b_value_dict = dict([(combined,avg_invalid_dispute_amount) for combined,avg_invalid_dispute_amount in zip(b_value['combined'],b_value['avg_invalid_dispute_amount'])])

b_value_dict = str(b_value_dict).replace("{","").replace("}","")
#
mean_amount_history_bsad = pd.read_csv('history2/normal_average.csv')


mean_amount_history_bsad = mean_amount_history_bsad[['FIN_KUNNR_hist_group','local_currency_mean']]

mean_amount_history_bsad = mean_amount_history_bsad.sort_values(by='FIN_KUNNR_hist_group')

#mean_amount_history['dispute_mean'] = np.round(mean_amount_history['dispute_mean'],5)

mean_invoice_amount_dict = dict([(kunnr,mean_amount) for kunnr,mean_amount in zip(mean_amount_history_bsad['FIN_KUNNR_hist_group'],
                                                                                  mean_amount_history_bsad['local_currency_mean'])])
mean_invoice_amount_dict = str(mean_invoice_amount_dict).replace("{","").replace("}","")

overall_amount_history_bsad = pd.read_csv('history2/total_history.csv')


overall_amount_history_bsad = overall_amount_history_bsad[['FIN_KUNNR_hist_group','total_local_currency_mean']]

overall_amount_history_bsad = overall_amount_history_bsad.sort_values(by='FIN_KUNNR_hist_group')

#mean_amount_history['dispute_mean'] = np.round(mean_amount_history['dispute_mean'],5)

overall_invoice_amount_dict = dict([(kunnr,mean_amount) for kunnr,mean_amount in zip(overall_amount_history_bsad['FIN_KUNNR_hist_group'],
                                                                                  overall_amount_history_bsad['total_local_currency_mean'])])
overall_invoice_amount_dict = str(overall_invoice_amount_dict).replace("{","").replace("}","")







train['ship_to'] = train['ship_to'].astype('str')







mapper = DataFrameMapper([

    (['deduction_created_date', 'customer_claim_date'], [CustomTransformFunctionGenerator(function="SubtractDatesPG")]

     ),
('product_category', [CategoricalDomain(invalid_value_treatment="as_is",
                                            missing_value_replacement="others"),
                          CustomTransformFunctionGenerator(function="HistoryXref",
                                                           arguments=xref_dict),

                          ]),


    (['payer', 'ship_to', 'product_category'],
     [CustomTransformFunctionGenerator(function="CalCustomerHistory",
                                       arguments=cal_customer_history_dict

                                       )],

     ),

    ('customer_claim_date', [
        CustomTransformFunctionGenerator(function="MonthOfYear")
    ]),

    ('ship_to', [
        CustomTransformFunctionGenerator(function="HistoryKunwe", arguments=kunwe_dict)
    ]),

    (['payer', 'original_dispute_amount'], [CustomTransformFunctionGenerator(function="HistoryAmount",
                                                                             arguments=mean_amount_dict)],

     ),

    (['payer', 'product_category'],
     [CustomTransformFunctionGenerator(function="RankInXrefHistory",
                                       arguments=category_rank_dict)],
     ),

    (['payer', 'original_dispute_amount'],
     [CustomTransformFunctionGenerator(function="HistoryBvalue",
                                       arguments=b_value_dict

                                       )],),

    (['payer', 'ship_to'],
     [CustomTransformFunctionGenerator(function="RankInKunweHistory",
                                       arguments=kunwe_rank_dict
                                       )],

     ),

(['cash_discount_local_currency','amount_local_currency'],
                           [CustomTransformFunctionGenerator(function='/')] ),

(['original_dispute_amount','amount_local_currency'],
                           [CustomTransformFunctionGenerator(function='/')] ),

(['payer','amount_local_currency_2'],
                           [CustomTransformFunctionGenerator(function="HistoryAmount",
                                                             arguments=mean_invoice_amount_dict
)], ),

(['payer','amount_local_currency'],
                           [CustomTransformFunctionGenerator(function="HistoryAmount",
                                                             arguments=overall_invoice_amount_dict

)], ),

])


model=RandomForestClassifier(n_estimators=50,random_state=9,max_depth=15,min_samples_split=40)
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()



#train_dataset_transformations = pd.DataFrame(mapper.fit_transform(train),columns=['create_minus_claim_date', 'category_history', 'cal_cust_history', 'ZZ_CLAIMDATE_SIMP_DT_month', 'ship_to_history', 'original_with_avg_dispute', 'rank_xref_in_kunnr', 'b_value', 'rank_kunwe_in_kunnr'])
#
#
# train_dataset_transformations['']
#
#train_dataset_transformations.to_csv('train_dataset_transformations.csv')

pipeline = PMMLPipeline([("mapper",mapper),("estimator",model)])
#
# train_transformations = pd.DataFrame(mapper.fit_transform(train),columns=['create_minus_claim_date', 'category_history', 'cal_cust_history', 'ZZ_CLAIMDATE_SIMP_DT_month', 'ship_to_history', 'original_with_avg_dispute', 'rank_xref_in_kunnr', 'b_value', 'rank_kunwe_in_kunnr'])
#
#
# train_transformations.to_csv('train_transformations.csv')

pipeline.fit(train,train['labels'])

test = pd.read_csv(r'test_set2.csv')
test['main_output_1']=(test['FIN_PAID_AMT']>(0.01*test['FIN_ORIGINAL_AMT']))
test['labels']=test['main_output_1'].map({True:-1,False:1})

test = test.rename(columns={'ZZ_CLAIMDATE_SIMP_DT': 'customer_claim_date', 'CREATE_TIME': 'deduction_created_date','ZZ_XREF3': 'product_category','KUNWE': 'ship_to','FIN_ORIGINAL_AMT': 'original_dispute_amount','FIN_KUNNR': 'payer','FIN_PAID_AMT': 'paid_amount'})
#

test['ship_to'] = test['ship_to'].astype('str').str.split('.').str[0]

#
# test_transformations = pd.DataFrame(mapper.fit_transform(test),columns=['create_minus_claim_date', 'category_history', 'cal_cust_history', 'ZZ_CLAIMDATE_SIMP_DT_month', 'ship_to_history', 'original_with_avg_dispute', 'rank_xref_in_kunnr', 'b_value', 'rank_kunwe_in_kunnr'])
#
# test_transformations.to_csv('test_transformations.csv')

test_result = pd.DataFrame()
test_result['output'] = pipeline.predict(test)
#
test_result['predict_proba1'] = pipeline.predict_proba(test)[:,0]
test_result['predict_proba2'] = pipeline.predict_proba(test)[:,1]

test_result['actual_result'] = test['labels']

from sklearn.metrics import classification_report

print(classification_report(test_result['actual_result'],test_result['output']))

from sklearn2pmml import sklearn2pmml
#
# #sklearn2pmml(pipeline, "only_b_value.pmml", user_classpath=[r"D:\jesus\sap\sklearn2pmml-plugin-1.0-SNAPSHOT.jar"],debug=True)
#
sklearn2pmml(pipeline, "pmmls\\pmml_with_all_features_disputes_1.pmml",user_classpath=[r"D:\jesus\sap\sklearn2pmml-plugin-1.0-SNAPSHOT.jar"], with_repr=True)


#
#
#
#
#
#
#
#
#
# #test = pd.read_csv(r'Data/UDM_DISPUTE_20171231-20180202.csv')
#
#
#
# test['KUNWE'] = test['KUNWE'].astype('str').str.split('.',0).str[0]
#
# test['main_output']=(test['FIN_PAID_AMT']>(0.01*test['FIN_ORIGINAL_AMT']))
# test['main_output']=test['main_output'].map({True:0,False:1})
# #
#
# #pd.DataFrame(data=test_dataset)
#
# test = test.rename(columns={'ZZ_CLAIMDATE_SIMP_DT': 'customer_claim_date', 'CREATE_TIME': 'deduction_created_date','ZZ_XREF3': 'product_category','KUNWE': 'ship_to','FIN_ORIGINAL_AMT': 'original_dispute_amount','FIN_KUNNR': 'payer','FIN_PAID_AMT': 'paid_amount'})
#
# test_dataset = pd.DataFrame(mapper.fit_transform(test),columns=['create_minus_claim_date', 'category_history', 'cal_cust_history', 'ZZ_CLAIMDATE_SIMP_DT_month', 'ship_to_history', 'original_with_avg_dispute', 'rank_xref_in_kunnr', 'b_value', 'rank_kunwe_in_kunnr'])
#
# x = pd.DataFrame(mapper.fit_transform(test),columns=['create_minus_claim_date', 'category_history', 'cal_cust_history', 'ZZ_CLAIMDATE_SIMP_DT_month', 'ship_to_history', 'original_with_avg_dispute', 'rank_xref_in_kunnr', 'b_value', 'rank_kunwe_in_kunnr'])
# x.to_csv('fixing_the_issue2.csv')
#
# from sklearn.linear_model import LogisticRegression
# test_result = pd.DataFrame()
# test_result['output'] = pipeline.predict(test)
# #
# test_result['predict_proba1'] = pipeline.predict_proba(test)[:,0]
# test_result['predict_proba2'] = pipeline.predict_proba(test)[:,1]
#
# test_result['actual_result'] = test['labels']
#
# from sklearn.metrics import classification_report
#
# print(classification_report(test_result['actual_result'],test_result['output']))
#
#
# #pipeline.predict()
# #
# # test_real = pd.read_csv('test_real3.csv',encoding='latin')
# #
# # test_results = pd.DataFrame(columns=['result','prob1','prob2'])
# #
# # test_results['result'] = pipeline.predict(test_real)
# #
# # from sklearn import metrics
# #
# # defe_test2 = metrics.classification_report(test_real['main_output'],test_results['result'])
# #
# # print(defe_test2)
# #
# # test_results[['prob1','prob2']] = pipeline.predict_proba(test_real)
# #
# # test_results['act_result'] = test_real['main_output']
# #
# # test_results.to_csv('pmml_test_results.csv')
#
#