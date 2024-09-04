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
