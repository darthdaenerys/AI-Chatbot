from chatbot import ChatBot
from models import Encoder,Decoder
import json
import time

with open('../hyperparameters.json','r') as f:
    data=json.load(f)

lstm_cells=data['lstm_cells']
embedding_dim=data['embedding_dim']
vocab_size=data['vocab_size']

encoder=Encoder(lstm_cells,embedding_dim,vocab_size)
decoder=Decoder(lstm_cells,embedding_dim,vocab_size)
chatbot=ChatBot(encoder,decoder)
chatbot.load_weights('../weights/weight')
text=''