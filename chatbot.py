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

    def build_inference_model(self,base_encoder,base_decoder):
        encoder_inputs=tf.keras.Input(shape=(None,))
        x=base_encoder.layers[0](encoder_inputs)
        x=base_encoder.layers[1](x)
        encoder_outputs,encoder_state_h,encoder_state_c=base_encoder.layers[2](x)
        encoder=tf.keras.models.Model(
            inputs=encoder_inputs,
            outputs=(encoder_outputs,encoder_state_h,encoder_state_c),name='chatbot_encoder'
        )

        decoder_inputs=tf.keras.Input(shape=(1,),name='decoder_inputs')
        all_encoder_inputs=tf.keras.Input(shape=(None,lstm_cells),name='all_encoder_inputs')
        decoder_input_state_h=tf.keras.Input(shape=(lstm_cells,),name='decoder_state_h_input')
        decoder_input_state_c=tf.keras.Input(shape=(lstm_cells,),name='decoder_state_c_input')
        x=base_decoder.layers[0](decoder_inputs)
        x=base_decoder.layers[2](x)
        context_vector,attention_weights=base_decoder.layers[1](decoder_input_state_h,all_encoder_inputs)
        context_vector=tf.expand_dims(context_vector,1)
        x=tf.concat([context_vector,x],axis=-1)
        decoder_outputs,decoder_state_h,decoder_state_c=base_decoder.layers[3](x,initial_state=[decoder_input_state_h,decoder_input_state_c])
        decoder_outputs=base_decoder.layers[-1](decoder_outputs)
        decoder=tf.keras.models.Model(
            inputs=[decoder_inputs,all_encoder_inputs,decoder_input_state_h,decoder_input_state_c],
            outputs=[decoder_outputs,decoder_state_h,decoder_state_c],name='chatbot_decoder'
        )
        return encoder,decoder

    def summary(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

    def softmax(self,z):
        return np.exp(z)/sum(np.exp(z))

    def sample(self,conditional_probability,temperature=0.8):
        conditional_probability = np.asarray(conditional_probability).astype("float64")
        conditional_probability = np.log(conditional_probability) / temperature
        reweighted_conditional_probability = self.softmax(conditional_probability)
        probas = np.random.multinomial(1, reweighted_conditional_probability, 1)
        return np.argmax(probas)

    def preprocess(self,text):
        text=clean_text(text)
        seq=np.zeros((1,max_sequence_length),dtype=np.int32)
        for i,word in enumerate(text.split()):
            seq[:,i]=sequences2ids(word,self.vectorize_layer).numpy()[0]
        return seq
    
    def call(self,text,config=None):
        input_seq=self.preprocess(text)
        encoder_outputs,state_h,state_c=self.encoder(input_seq,training=False)
        target_seq=np.zeros((1,1))
        target_seq[:,:]=sequences2ids(['<sos>'],self.vectorize_layer).numpy()[0][0]
        stop_condition=False
        decoded=[]
        
        print('<sos>',end='',flush=True)
        time.sleep(.3)
        
        while not stop_condition:
            
            decoder_outputs,state_h,state_c=self.decoder([target_seq,encoder_outputs,state_h,state_c],training=False)
            # index=tf.argmax(decoder_outputs[:,-1,:],axis=-1).numpy().item()
            index=self.sample(decoder_outputs[0,0,:]).item()
            word=ids2sequences([index],self.vectorize_layer)
            word=word.strip()
            
            if word=='<eos>' or len(decoded)>=max_sequence_length:
                time.sleep(.3)
                print('<eos>',end='',flush=True)
                self.eos_recieved=True
                stop_condition=True
                
            else:
                if word in [".,!?/&$:;*"]:
                    print(word,end='',flush=True)
                else:
                    if self.first_word:
                        word=word[0].upper()+word[1:]
                        self.first_word=False
                    print(' '+word,end='',flush=True)
                    
                decoded.append(index)
                target_seq=np.zeros((1,1))
                target_seq[:,:]=index
    
    def prompt(self,text):
        self.first_word=True
        self(text)
        
        if not self.eos_recieved:
            time.sleep(.3)
            print('<eos>',end='',flush=True)
            

