import tensorflow as tf
import numpy as np
from utilities.utils import *
from utilities.process import *
from utilities.vocab import get_vocabulary
import time

class ChatBot(tf.keras.models.Model):
    def __init__(self,base_encoder,base_decoder,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.vectorize_layer=get_vocabulary()
        self.encoder,self.decoder=self.build_inference_model(base_encoder,base_decoder)
        self.first_word=True
        self.eos_recieved=False