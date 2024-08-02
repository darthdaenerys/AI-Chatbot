import pandas as pd
import tensorflow as tf
import re
import os
import json
from tensorflow.keras.layers import TextVectorization
from utilities.utils import *
from utilities.plot import *

with open('../hyperparameters.json','r') as f:
    data=json.load(f)

lstm_cells=data['lstm_cells']
embedding_dim=data['embedding_dim']
vocab_size=data['vocab_size']
max_sequence_length=data['max_sequence_length']
buffer_size=data['buffer_size']
batch_size=data['batch_size']

def load_data():
    df=pd.read_csv('../data/dialogue.txt',sep='\t',names=['question','answer'],encoding='utf-8')
    df2=pd.read_csv(os.path.join('../data','chatbot dataset.txt'),names=['question','answer'],sep='\t',encoding='utf-8')
    df=pd.concat([df,df2],axis=0)
    df2=pd.read_csv(os.path.join('../data','dialogs_expanded.csv'),usecols=['question','answer'],encoding='utf-8',sep='\t')
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

    vectorize_layer=TextVectorization(
        max_tokens=vocab_size,
        standardize=None,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )
    vectorize_layer.adapt(df['encoder_inputs']+' '+df['decoder_targets']+' <sos> <eos>',batch_size=512)

    x=sequences2ids(df['encoder_inputs'],vectorize_layer)
    yd=sequences2ids(df['decoder_inputs'],vectorize_layer)
    y=sequences2ids(df['decoder_targets'],vectorize_layer)

    data=tf.data.Dataset.from_tensor_slices((x,yd,y))
    data=data.shuffle(buffer_size)

    train_data=data.take(int(.9*len(data)))
    train_data=train_data.cache()
    train_data=train_data.shuffle(buffer_size)
    train_data=train_data.batch(batch_size)
    train_data=train_data.prefetch(tf.data.AUTOTUNE)
    train_data_iterator=train_data.as_numpy_iterator()

    val_data=data.skip(int(.9*len(data))).take(int(.1*len(data)))
    val_data=val_data.batch(batch_size)
    val_data=val_data.prefetch(tf.data.AUTOTUNE)

    _=train_data_iterator.next()
    return train_data,val_data,train_data_iterator