from trainer import ChatBotTrainer
import tensorflow as tf
from models import Encoder,Decoder
from utilities.process import *
import json

with open('hyperparameters.json','r') as f:
    data=json.load(f)

lstm_cells=data['lstm_cells']
embedding_dim=data['embedding_dim']
vocab_size=data['vocab_size']
max_sequence_length=data['max_sequence_length']
buffer_size=data['buffer_size']
batch_size=data['batch_size']
learning_rate=data['learning_rate']

df=load_data()
train_data,val_data,train_data_iterator=preprocess(df)

encoder=Encoder(lstm_cells,embedding_dim,vocab_size,name='encoder')
decoder=Decoder(lstm_cells,embedding_dim,vocab_size,name='decoder')

model=ChatBotTrainer(encoder,decoder,name='chatbot_trainer')