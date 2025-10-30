# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable(package="SemEnr")
class MulDimLayer(Layer):
    def __init__(self, **kwargs):
        super(MulDimLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3, "MulDimLayer expects 3D input: (batch, time, dim)"
        self.dim = int(input_shape[2])
        super(MulDimLayer, self).build(input_shape)


    def call(self, inputs):
        return inputs * (self.dim ** 0.5)


    def compute_output_shape(slef, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


    def get_config(self):
        cfg = super().get_config()
        return cfg





