# from keras.engine import Layer
from tensorflow.keras.layers import Layer

# from keras import initializers
from tensorflow.keras import initializers

# from keras import backend as K
import tensorflow as tf

# Attention GRU network       
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        # x = K.permute_dimensions(inputs, (0, 2, 1))
        x = tf.transpose(inputs, perm=(0,2,1))


        # x.shape = (batch_size, seq_len, time_steps)
        # a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        a = tf.nn.softmax(tf.math.tanh(tf.linalg.matmul(x, self.W) + self.b), axis=-1)

        # outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = tf.transpose(a * x, perm=(0,2,1))

        # outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


