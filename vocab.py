from process import load_data,clean_text
from tensorflow.keras.layers import TextVectorization
from hyperparameters import *

def get_vocabulary():
    df=load_data()
    df['question tokens']=df['question'].apply(lambda x:len(x.split()))
    df['answer tokens']=df['answer'].apply(lambda x:len(x.split()))
    df.drop(columns=['answer tokens','question tokens'],axis=1,inplace=True)
    df['encoder_inputs']=df['question'].apply(clean_text)
    df['decoder_targets']=df['answer'].apply(clean_text)+' <eos>'
    df['decoder_inputs']='<sos> '+df['answer'].apply(clean_text)+' <eos>'
    df['encoder input tokens']=df['encoder_inputs'].apply(lambda x:len(x.split()))
    df['decoder input tokens']=df['decoder_inputs'].apply(lambda x:len(x.split()))
    df['decoder target tokens']=df['decoder_targets'].apply(lambda x:len(x.split()))
    df.drop(columns=['question','answer','encoder input tokens','decoder input tokens','decoder target tokens'],axis=1,inplace=True)

    vectorize_layer=TextVectorization(
        max_tokens=vocab_size,
        standardize=None,
        output_mode='int',
        output_sequence_length=max_sequence_length
    )
    vectorize_layer.adapt(df['encoder_inputs']+' '+df['decoder_targets']+' <sos> <eos>',batch_size=512)
    return vectorize_layer