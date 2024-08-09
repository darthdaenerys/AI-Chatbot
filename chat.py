from chatbot import ChatBot
from models import Encoder,Decoder
import json
import time

with open('../hyperparameters.json','r') as f:
    data=json.load(f)