import pandas as pd
import tensorflow as tf
import re
import os
from hyperparameters import *
from tensorflow.keras.layers import TextVectorization
from utils import *
from plot import *

def load_data():
    df=pd.read_csv('data/dialogue.txt',sep='\t',names=['question','answer'],encoding='utf-8')
    df2=pd.read_csv(os.path.join('data','chatbot dataset.txt'),names=['question','answer'],sep='\t',encoding='utf-8')
    df=pd.concat([df,df2],axis=0)
    df2=pd.read_csv(os.path.join('data','dialogs_expanded.csv'),usecols=['question','answer'],encoding='utf-8')
    df=pd.concat([df,df2],axis=0)
    del df2
    return df

def clean_text(text):
    text=re.sub('-',' ',text.lower())
    text=re.sub('[.]',' . ',text)
    text=re.sub('[1]',' 1 ',text)
    text=re.sub('[2]',' 2 ',text)
    text=re.sub('[3]',' 3 ',text)
    text=re.sub('[4]',' 4 ',text)
    text=re.sub('[5]',' 5 ',text)
    text=re.sub('[6]',' 6 ',text)
    text=re.sub('[7]',' 7 ',text)
    text=re.sub('[8]',' 8 ',text)
    text=re.sub('[9]',' 9 ',text)
    text=re.sub('[0]',' 0 ',text)
    text=re.sub('[,]',' , ',text)
    text=re.sub('[?]',' ? ',text)
    text=re.sub('[!]',' ! ',text)
    text=re.sub('[$]',' $ ',text)
    text=re.sub('[&]',' & ',text)
    text=re.sub('[/]',' / ',text)
    text=re.sub('[:]',' : ',text)
    text=re.sub('[;]',' ; ',text)
    text=re.sub('[*]',' * ',text)
    text=re.sub('[\']',' \' ',text)
    text=re.sub('[\"]',' \" ',text)
    text=re.sub('\t',' ',text)
    return text

def preprocess(df):
    df['question tokens']=df['question'].apply(lambda x:len(x.split()))
    df['answer tokens']=df['answer'].apply(lambda x:len(x.split()))
    show_qna_tokens(df)
    df.drop(columns=['answer tokens','question tokens'],axis=1,inplace=True)
    df['encoder_inputs']=df['question'].apply(clean_text)
    df['decoder_targets']=df['answer'].apply(clean_text)+' <eos>'
    df['decoder_inputs']='<sos> '+df['answer'].apply(clean_text)+' <eos>'
    df['encoder input tokens']=df['encoder_inputs'].apply(lambda x:len(x.split()))
    df['decoder input tokens']=df['decoder_inputs'].apply(lambda x:len(x.split()))
    df['decoder target tokens']=df['decoder_targets'].apply(lambda x:len(x.split()))
    show_encdec_tokens(df)
    df.drop(columns=['question','answer','encoder input tokens','decoder input tokens','decoder target tokens'],axis=1,inplace=True)