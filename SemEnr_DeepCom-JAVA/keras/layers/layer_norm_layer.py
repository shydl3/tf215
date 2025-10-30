# -*- coding: utf-8 -*-
# TensorFlow 2.15 等价实现：不改动原有功能/形状/计算顺序
import tensorflow as tf
from tensorflow.keras.layers import Layer

class LayerNormLayer(Layer):
    def __init__(self,
                 center=True,
                 scale=False,
                 epsilon=None,
                 **kwargs):
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma, self.beta = 0., 0.  # 原文件中存在但未使用，保留以保持兼容
        super(LayerNormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 与原实现一致：两个 (feat, feat) 的权重，初始化器同为 'uniform'
        self.W_1 = self.add_weight(
            name='w_1',
            shape=(input_shape[2], input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        self.W_2 = self.add_weight(
            name='w_2',
            shape=(input_shape[2], input_shape[2]),
            initializer='uniform',
            trainable=True
        )
        super(LayerNormLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 与原逻辑严格等价：逐特征做均值/方差 → 标准化 → 两层仿射+ReLU → Dropout → 残差
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)  # epsilon 仍沿用原默认值行为
        outputs_1 = (inputs - mean) / std
        outputs_2 = tf.nn.relu(tf.linalg.matmul(outputs_1, self.W_1))
        outputs_3 = tf.linalg.matmul(outputs_2, self.W_2)
        outputs_4 = tf.nn.dropout(outputs_3, rate=0.25)  # K.dropout(level=0.25) 等价
        outputs = outputs_1 + outputs_4

        print(outputs)  # 保留原有打印副作用
        return outputs

    def compute_output_shape(self, input_shape):
        # 保持原返回契约
        return (input_shape[0], input_shape[1], input_shape[2])
