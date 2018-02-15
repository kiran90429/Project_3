import numpy as np
import ast
import pandas as pd

def CalCustomerHistory(X,arguments=None):
    arguments= "{" + arguments + "}"

    print(arguments)

    history_mappings= ast.literal_eval(arguments)

    history_dataframe = pd.DataFrame(list(history_mappings.items()),columns=['kunnr|kunwe|xref','history'])

    history_df = history_dataframe['kunnr|kunwe|xref'].apply(lambda x:pd.Series(x.split('|')))

    history_df['history'] = history_dataframe['history']

    history_df.columns=['kunnr','kunwe','xref','history']

    history_df['kunnr'] = history_df['kunnr'].str.split('.',0).str[0]

    history_df['kunwe'] = history_df['kunwe'].str.split('.',0).str[0]

    history_df['3key'] = history_df['kunnr'] + history_df['kunwe'] + history_df['xref']

    history_df['2key'] = history_df['kunnr'] + history_df['kunwe']

    history_df['1key'] = history_df['kunnr']

    history_df_for_join_1 = history_df[(history_df['kunnr'] != 'others') & (history_df['kunwe'] != 'others') & (history_df['xref'] != 'others')]

    history_df_for_join_2 = history_df[(history_df['xref'] == 'others') & (history_df['kunwe'] != 'others')]

    history_df_for_join_3 = history_df[(history_df['xref'] == 'others') & (history_df['kunwe'] == 'others') & (history_df['kunnr'] != 'others')]

    history_df_for_join_4 = history_df[(history_df['xref'] == 'others') & (history_df['kunwe'] == 'others') & (history_df['kunnr'] == 'others')]

    train_dataframe = pd.DataFrame(X.tolist(),columns=['kunnr','kunwe','xref'])

    train_dataframe['kunnr'] = train_dataframe['kunnr'].astype(str)

    train_dataframe['kunwe'].fillna("others",inplace=True)

    train_dataframe['kunwe'] = train_dataframe['kunwe'].str.split('.', 0).str[0]

    train_dataframe['kunwe'].replace('nan',"others",inplace=True)

    # train_dataframe['kunwe'] = train_dataframe['kunwe'].astype(int).astype(str)
    #
    # train_dataframe['kunwe'].replace('0','others',inplace=True)
    train_dataframe['3key'] = train_dataframe['kunnr'] + train_dataframe['kunwe'] + train_dataframe['xref']

    train_dataframe['2key'] = train_dataframe['kunnr'] + train_dataframe['kunwe']

    train_dataframe['1key'] = train_dataframe['kunnr']

    train_dataframe['index_final_history'] = range(1, len(train_dataframe) + 1)

    data_join_1 = pd.merge(train_dataframe, history_df_for_join_1, on='3key', how='inner')

    train_data_for_join_2 = train_dataframe[~train_dataframe['index_final_history'].isin(data_join_1['index_final_history'])]

    data_join_2 = pd.merge(train_data_for_join_2, history_df_for_join_2, on='2key', how='inner')

    train_data_for_join_3 = train_data_for_join_2[
        ~train_data_for_join_2['index_final_history'].isin(data_join_2['index_final_history'])]

    data_join_3 = pd.merge(train_data_for_join_3, history_df_for_join_3, on='1key', how='inner')

    train_data_for_join_4 = train_data_for_join_3[
        ~train_data_for_join_3['index_final_history'].isin(data_join_3['index_final_history'])]

    train_data_for_join_4['history'] = history_df_for_join_4['history'].iloc[0]

    data_join_4 = pd.merge(train_data_for_join_4, history_df_for_join_4, on='history')

    x = data_join_3.columns.tolist()

    y = data_join_4.columns.tolist()

    z = set(x).intersection(set(y))

    final_columns = list(z)

    x1 = data_join_1.columns
    y1 = data_join_2.columns

    z1 = set(x1).intersection(y1)

    final_columns = z.intersection(z1)

    final_columns = list(final_columns)

    final_train_joined = pd.concat([data_join_1[final_columns], data_join_2[final_columns], data_join_3[final_columns],
                                    data_join_4[final_columns]])


    print(final_train_joined['history'].values)
    final_train_joined.to_csv('cal_customer_history.csv')

    return np.round(final_train_joined['history'].values,4)
