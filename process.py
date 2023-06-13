import pandas as pd
import os

def load_data():
    df=pd.read_csv('data/dialogue.txt',sep='\t',names=['question','answer'],encoding='utf-8')
    df2=pd.read_csv(os.path.join('data','chatbot dataset.txt'),names=['question','answer'],sep='\t',encoding='utf-8')
    df=pd.concat([df,df2],axis=0)
    df2=pd.read_csv(os.path.join('data','dialogs_expanded.csv'),usecols=['question','answer'],encoding='utf-8')
    df=pd.concat([df,df2],axis=0)
    del df2
    return df
    