import tensorflow as tf

class ChatBotTrainer(tf.keras.models.Model):
    def __init__(self,encoder,decoder,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.encoder=encoder
        self.decoder=decoder

    def loss_fn(self,y_true,y_pred):
        loss=self.loss(y_true,y_pred)
        mask=tf.math.logical_not(tf.math.equal(y_true,0))
        mask=tf.cast(mask,dtype=loss.dtype)
        loss*=mask
        return tf.reduce_mean(loss)
            
    def correct_fn(self,y_true,y_pred):
        pred_values = tf.cast(tf.argmax(y_pred, axis=-1), dtype='int64')
        correct = tf.cast(tf.equal(y_true, pred_values), dtype='float64')
        mask = tf.cast(tf.greater(y_true, 0),dtype='float64')
        n_correct = tf.keras.backend.sum(mask*correct)
        n_total = tf.keras.backend.sum(mask)
        return n_correct,n_total

    def __call__(self,inputs):
        encoder_inputs,decoder_inputs=inputs
        encoder_outputs,decoder_state_h,decoder_state_c=self.encoder(encoder_inputs)
        all_outputs=[]
        for t in range(decoder_inputs.shape[-1]):
            decoder_outputs,decoder_state_h,decoder_state_c=self.decoder(decoder_inputs[:,t:t+1],encoder_outputs,decoder_state_h,decoder_state_c)
            all_outputs.append(decoder_outputs)
        decoder_outputs=tf.keras.layers.Lambda(lambda x:tf.keras.backend.concatenate(x,axis=1))(all_outputs)
        return decoder_outputs

    def train_step(self,batch):
        encoder_inputs,decoder_inputs,y=batch
        loss=0
        correct=0
        total=0
        with tf.GradientTape() as tape:
            encoder_outputs,decoder_state_h,decoder_state_c=self.encoder(encoder_inputs,training=True)
            for t in range(y.shape[1]):
                decoder_output,decoder_state_h,decoder_state_c=self.decoder(decoder_inputs[:,t:t+1],encoder_outputs,decoder_state_h,decoder_state_c,training=True)
                decoder_output=tf.squeeze(decoder_output,axis=1)
                loss+=self.loss_fn(y[:,t],decoder_output)
                correct_,total_=self.correct_fn(y[:,t],decoder_output)
                correct+=correct_
                total+=total_
        batch_loss=loss/y.shape[1]
        acc=correct/total
        variables=self.encoder.trainable_variables+self.decoder.trainable_variables
        grads=tape.gradient(loss,variables)
        self.optimizer.apply_gradients(zip(grads,variables))
        metrics={'loss':batch_loss,'accuracy':acc}
        return metrics