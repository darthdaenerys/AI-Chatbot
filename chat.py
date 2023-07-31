from chatbot import ChatBot
from models import Encoder,Decoder
from hyperparameters import *

encoder=Encoder(lstm_cells,embedding_dim,vocab_size)
decoder=Decoder(lstm_cells,embedding_dim,vocab_size)
chatbot=ChatBot(encoder,decoder)
chatbot.load_weights('weights/weights')
text=''

print(f'Bot: Hello, I\'m an AI chatbot.')
while True:
    print(f'User: ',end='')
    text=input()
    if text.lower()=='quit' or text.lower()=='exit':
        print('\n*** Exiting ***')
        break
    reply=chatbot.prompt(text)
    print(f'Bot: {reply}\n')