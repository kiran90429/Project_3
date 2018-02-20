import numpy as np
import ast
import pandas as pd

def HistoryBvalue(X,arguments=None):
    arguments= "{" + arguments + "}"

    print(arguments)

    history_mappings= ast.literal_eval(arguments)

    history_dataframe = pd.DataFrame(list(history_mappings.items()),columns=['kunnr|value_label','avg_invalid_dispute_amount'])

    history_df = history_dataframe['kunnr|value_label'].apply(lambda x:pd.Series(x.split('|')))

    history_df['avg_invalid_dispute_amount'] = history_dataframe['avg_invalid_dispute_amount']

    history_df.columns=['kunnr','value_label','avg_invalid_dispute_amount']

    history_df['kunnr'] = history_df['kunnr'].str.split('.',0).str[0]

    #history_df['kunwe'] = history_df['kunwe'].str.split('.',0).str[0]

    train_dataframe = pd.DataFrame(X.tolist(),columns=['kunnr','dispute_amount'])

    train_dataframe['kunnr'] = train_dataframe['kunnr'].astype(str)

    train_dataframe['kunnr'] = train_dataframe['kunnr'].str.split('.', 0).str[0]

    train_dataframe['kunnr'] = train_dataframe['kunnr'].apply(
        lambda x: x if x in history_df['kunnr'].unique().tolist() else '999999999')

    train_dataframe = pd.merge(train_dataframe, history_df, on='kunnr', how='left')
    train_dataframe['b_value'] = train_dataframe['dispute_amount'] / train_dataframe['avg_invalid_dispute_amount']
    train_dataframe.loc[train_dataframe['value_label'] == 'LOW', 'b_value'] = 1 / train_dataframe.loc[train_dataframe['value_label'] == 'LOW', 'b_value']

    train_dataframe.loc[train_dataframe['b_value'] <= 0.005, 'b_value'] = 0.005
    train_dataframe.loc[train_dataframe['b_value'] >= 5, 'b_value'] = 5
    train_dataframe.loc[train_dataframe['b_value'] <= 0.005, 'b_value'] = 0.005
    train_dataframe.loc[train_dataframe['b_value'] >= 5, 'b_value'] = 5


    print(train_dataframe['b_value'].values)

    train_dataframe.to_csv('pmml_b_value.csv')

    return train_dataframe['b_value'].values
