# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Layer

class JointSelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(JointSelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called on a list of 2 inputs.')
        if input_shape[0][2] != input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size') 

        self.dim = input_shape[0][2]

        self.W_qc = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer='glorot_uniform',
            name='W_qc',
            trainable=True,
        )
        self.W_vc = self.add_weight(
                shape=(input_shape[0][2], input_shape[0][2]),
                initializer='glorot_uniform',
                name='W_vc',
                trainable=True,
            )
        self.W_kd = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer='glorot_uniform',
            name='W_kd',
            trainable=True,
        )
        self.W_vd = self.add_weight(
            shape=(input_shape[0][2], input_shape[0][2]),
            initializer='glorot_uniform',
            name='W_vd',
            trainable=True,
        )

        super(JointSelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 等价替换：K.dot -> tf.linalg.matmul
        Q_c = tf.linalg.matmul(inputs[0], self.W_qc)  # (B, T_c, dim)
        V_c = tf.linalg.matmul(inputs[0], self.W_vc)  # (B, T_c, dim)
        K_d = tf.linalg.matmul(inputs[1], self.W_kd)  # (B, T_d, dim)
        V_d = tf.linalg.matmul(inputs[1], self.W_vd)  # (B, T_d, dim)

        # 等价替换：K.permute_dimensions -> tf.transpose
        K_d_T = tf.transpose(K_d, perm=(0, 2, 1))     # (B, dim, T_d)

        # 等价替换：K.batch_dot -> tf.linalg.matmul（按批维做乘法）
        # 原逻辑：A = softmax(Q_c @ K_d^T) / sqrt(dim)  —— 注意保持**先 softmax 后缩放**的顺序
        logits = tf.linalg.matmul(Q_c, K_d_T)         # (B, T_c, T_d)
        A = tf.nn.softmax(logits, axis=-1) / (self.dim ** 0.5)  # (B, T_c, T_d)

        # c = A @ V_d, d = A^T @ V_c
        c = tf.linalg.matmul(A, V_d)                  # (B, T_c, dim)
        d = tf.linalg.matmul(tf.transpose(A, perm=(0, 2, 1)), V_c)  # (B, T_d, dim)

        # 等价替换：K.mean(axis=1, keepdims=False) -> tf.reduce_mean(keepdims=False)
        C = tf.reduce_mean(c, axis=1, keepdims=False)  # (B, dim)
        D = tf.reduce_mean(d, axis=1, keepdims=False)  # (B, dim)

        return [C, D]


    def compute_output_shape(self, input_shape):
        return [(None, input_shape[0][2]), (None, input_shape[1][2])]