import tensorflow as tf
from vocab import get_vocabulary

class ChatBot(tf.keras.models.Model):
    def __init__(self,base_encoder,base_decoder,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.vectorize_layer=get_vocabulary()
        self.encoder,self.decoder=self.build_inference_model(base_encoder,base_decoder)