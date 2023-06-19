from trainer import ChatBotTrainer
import tensorflow as tf
from models import Encoder,Decoder
from hyperparameters import *
from process import *

df=load_data()
train_data,val_data,train_data_iterator=preprocess(df)

encoder=Encoder(lstm_cells,embedding_dim,vocab_size,name='encoder')
decoder=Decoder(lstm_cells,embedding_dim,vocab_size,name='decoder')

model=ChatBotTrainer(encoder,decoder,name='chatbot_trainer')
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    weighted_metrics=['loss','accuracy']
)

history=model.fit(
    train_data,
    epochs=2,
    validation_data=val_data,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('checkpoint/ckpt',verbose=1,save_best_only=True,save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
)
model.load_weights('checkpoint/ckpt')
model.save_weights('weights/weights')
encoder.save_weights('weights/encoder')
decoder.save_weights('weights/decoder')