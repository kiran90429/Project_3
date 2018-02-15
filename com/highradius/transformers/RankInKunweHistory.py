import numpy as np
import ast
import pandas as pd

def RankInKunweHistory(X,arguments=None):
    arguments= "{" + arguments + "}"

    print(arguments)

    rank_mappings = ast.literal_eval(arguments)

    rank_dataframe = pd.DataFrame(list(rank_mappings.items()),columns=['kunnr|kunwe|kunwe_label','rank'])

    rank_df = rank_dataframe['kunnr|kunwe|kunwe_label'].apply(lambda x:pd.Series(x.split('|')))

    rank_df['rank'] = rank_dataframe['rank']

    rank_df.columns=['kunnr','kunwe','kunwe_label','rank']

    rank_df['kunnr'] = rank_df['kunnr'].str.split('.',0).str[0]

    rank_df['kunwe'] = rank_df['kunwe'].str.split('.',0).str[0]

    train_dataframe = pd.DataFrame(X.tolist(),columns=['kunnr','kunwe'])

    #train_dataframe['kunnr'] = train_dataframe['kunnr'].astype('str')


    train_dataframe['kunnr'] = train_dataframe['kunnr'].astype('int').apply(lambda x:str(x))

    train_dataframe['kunnr'] = train_dataframe['kunnr'].apply(lambda x: x if x in rank_df['kunnr'].unique().tolist() else 999999999)

    #train_dataframe['kunwe'].fillna(0,inplace=True)


    train_dataframe['kunwe'].fillna("others",inplace=True)

    train_dataframe['kunwe'] = train_dataframe['kunwe'].str.split('.', 0).str[0]

    train_dataframe['kunwe'].replace('nan',"others",inplace=True)

    train_dataframe['kunwe'] = train_dataframe['kunwe'].apply(lambda x:str(x))

    rank_df['kunnr'] = rank_df['kunnr'].astype('str')
    rank_df['kunwe'] = rank_df['kunwe'].astype('str')
    rank_df['kunwe_label'] = rank_df['kunwe_label'].astype('str')
    train_dataframe['kunnr'] = train_dataframe['kunnr'].astype('str')
    train_dataframe['kunwe'] = train_dataframe['kunwe'].astype('str')
    train_dataframe = pd.merge(train_dataframe, rank_df, how='left', on=['kunnr', 'kunwe'])
    del train_dataframe['rank']
    train_dataframe.loc[train_dataframe['kunwe_label'].isnull(), 'kunwe_label'] = 'others'
    train_dataframe = pd.merge(train_dataframe, rank_df.drop_duplicates(['kunnr', 'kunwe_label']),
                     on=['kunnr', 'kunwe_label'], how='left')
    train_dataframe.rename(columns={'kunwe_x':'kunwe'},inplace=True)
    print(train_dataframe['rank'].values)
    train_dataframe.to_csv('rank_in_kunwe.csv')
    return train_dataframe['rank'].values

