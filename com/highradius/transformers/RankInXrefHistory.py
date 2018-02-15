import numpy as np
import ast
import pandas as pd

def RankInXrefHistory(X,arguments=None):
    arguments= "{" + arguments + "}"

    print(arguments)

    rank_mappings = ast.literal_eval(arguments)

    rank_dataframe = pd.DataFrame(list(rank_mappings.items()),columns=['kunnr|xref|xref_label','rank'])

    rank_df = rank_dataframe['kunnr|xref|xref_label'].apply(lambda x:pd.Series(x.split('|')))

    rank_df['rank'] = rank_dataframe['rank']

    rank_df.columns=['kunnr','xref','xref_label','rank']

    rank_df['kunnr'] = rank_df['kunnr'].str.split('.',0).str[0]

    #rank_df['xref'] = rank_df['xref'].str.split('.',0).str[0]

    train_dataframe = pd.DataFrame(X.tolist(),columns=['kunnr','xref'])

    #train_dataframe['kunnr'] = train_dataframe['kunnr'].astype('str')


    train_dataframe['kunnr'] = train_dataframe['kunnr'].astype('int').apply(lambda x:str(x))

    train_dataframe['kunnr'] = train_dataframe['kunnr'].apply(lambda x: x if x in rank_df['kunnr'].unique().tolist() else '999999999')

    #train_dataframe['kunwe'] = train_dataframe['kunwe'].astype('int').apply(lambda x:str(x))
    #joined
    train_dataframe = pd.merge(train_dataframe, rank_df, how='left', on=['kunnr', 'xref'])
    del train_dataframe['rank']
    train_dataframe.loc[train_dataframe['xref_label'].isnull(), 'xref_label'] = 'others'
    train_dataframe = pd.merge(train_dataframe, rank_df.drop_duplicates(['kunnr', 'xref_label']),
                     on=['kunnr', 'xref_label'], how='left')
    print(train_dataframe['rank'].values)

    train_dataframe.to_csv('rank_in_xref.csv')
    return train_dataframe['rank'].values

