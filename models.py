import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,LayerNormalization

class Encoder(tf.keras.models.Model):
    def __init__(self,units,embedding_dim,vocab_size,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.units=units
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.embedding=Embedding(
            vocab_size,
            embedding_dim,
            name='encoder_embedding',
            mask_zero=True,
            embeddings_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.normalize=LayerNormalization()
        self.lstm=LSTM(
            units,
            dropout=.3,
            return_state=True,
            return_sequences=True,
            name='encoder_lstm',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
    
    def call(self,encoder_inputs):
        self.inputs=encoder_inputs
        x=self.embedding(encoder_inputs)
        x=self.normalize(x)
        x=Dropout(.3)(x)
        encoder_outputs,encoder_state_h,encoder_state_c=self.lstm(x)
        self.outputs=[encoder_outputs,encoder_state_h,encoder_state_c]
        return encoder_outputs,encoder_state_h,encoder_state_c

class Decoder(tf.keras.models.Model):
    def __init__(self,units,embedding_dim,vocab_size,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.units=units
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embedding=Embedding(
            vocab_size,
            embedding_dim,
            name='decoder_embedding',
            mask_zero=True,
            embeddings_initializer=tf.keras.initializers.HeNormal()
        )
        self.attention=BahdanauAttention(units)
        self.normalize=LayerNormalization()
        self.lstm=LSTM(
            units,
            dropout=.3,
            return_state=True,
            return_sequences=True,
            name='decoder_lstm',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        self.fc=Dense(
            vocab_size,
            activation='softmax',
            name='decoder_dense',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
    
    def call(self,decoder_inputs,encoder_outputs,decoder_state_h,decoder_state_c):
        # decoder_inputs: (batch size,1)
        x=self.embedding(decoder_inputs)
        x=self.normalize(x)
        # x: (batch size,1,embedding dim)
        context_vector, attention_weights = self.attention(decoder_state_h,encoder_outputs)
        context_vector=tf.expand_dims(context_vector,1)
        # context vector: (batch size,1,units)
        x=tf.concat([context_vector,x],axis=-1)
        # x: (batch size,1,embedding_dim+units)
        decoder_outputs,decoder_state_h,decoder_state_c=self.lstm(x,initial_state=[decoder_state_h,decoder_state_c])
        # decoder outputs: (batch size,1,vocab size)
        # decoder state: (batch size, units)
        x=Dropout(.2)(decoder_outputs)
        x=self.fc(x)
        return x,decoder_state_h,decoder_state_c