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