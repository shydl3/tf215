
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

@tf.keras.utils.register_keras_serializable(package="SemEnr")
class PositionEncodingLayer(Layer):
    def __init__(self, **kwargs):
        super(PositionEncodingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[2]
        self.n_words = input_shape[1]
        super(PositionEncodingLayer, self).build(input_shape)

    def call(self, inputs):
        # 原实现：调用 numpy 版本的 positional_encoding，然后与 inputs 相加
        # 保持行为一致：将 numpy 结果转换为与 Keras floatx 一致的张量
        angle_rates = positional_encoding(int(self.n_words), int(self.dim))
        angle_rates = tf.convert_to_tensor(angle_rates, dtype=tf.keras.backend.floatx())
        return inputs + angle_rates  # 广播相加，形状 (1, T, D) + (B, T, D)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


def positional_encoding(pos, d_model):
    # 与原实现等价的 numpy 版本：返回形状 (1, pos, d_model)
    def get_angles(position, i):
        return position / np.power(10000., 2. * (i // 2.) / np.float32(d_model))
    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return angle_rates[np.newaxis, ...]


@tf.keras.utils.register_keras_serializable(package="SemEnr")
class PositionEmbedding(Layer):
    """可训练位置Embedding：支持 add / mul / concat、hierarchical、自定义position ids"""
    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 hierarchical=None,
                 embeddings_initializer='zeros',
                 custom_position_ids=False,
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """
        如果 custom_position_ids=True，则 inputs 为 [inputs, position_ids]
        否则 inputs 为张量。
        """
        # 与原语义一致, 先解包，再取 shape
        if self.custom_position_ids:
            inputs, position_ids = inputs
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        if self.custom_position_ids:
            if position_ids.dtype != tf.int32:
                position_ids = tf.cast(position_ids, tf.int32)
        else:
            # [[0,1,2,...,seq_len-1]] 形状 (1, seq_len)
            position_ids = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1.0 - alpha)
            # 两段分层位置编码
            embeddings_x = tf.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = tf.gather(embeddings, position_ids % self.input_dim)
            pos_embeddings = alpha * embeddings_x + (1.0 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                pos_embeddings = tf.gather(self.embeddings, position_ids)
            else:
                pos_embeddings = self.embeddings[tf.newaxis, :seq_len, :]

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        elif self.merge_mode == 'mul':
            return inputs * pos_embeddings
        else:
            # concat 分支：如果不是 custom ids，需要按 batch 扩展
            if not self.custom_position_ids:
                pos_embeddings = tf.tile(pos_embeddings, [batch_size, 1, 1])
            return tf.concat([inputs, pos_embeddings], axis=-1)

    def compute_output_shape(self, input_shape):
        # 保持与原实现一致输出
        if self.custom_position_ids:
            input_shape = input_shape[0]
        if self.merge_mode in ['add', 'mul']:
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.output_dim)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base = super(PositionEmbedding, self).get_config()
        return dict(list(base.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="SemEnr")
class SinusoidalPositionEmbedding(Layer):
    """Sin-Cos 位置Embedding：支持 add / mul / concat、自定义 position ids"""
    def __init__(self, output_dim, merge_mode='add', custom_position_ids=False, **kwargs):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """
        如果 custom_position_ids=True，则 inputs 为 [inputs, position_ids]
        否则 inputs 为张量。
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        if not self.custom_position_ids:
            # [[0,1,2,...,seq_len-1]]，float类型，形状 (1, seq_len)
            position_ids = tf.range(0, seq_len, dtype=tf.keras.backend.floatx())[
                tf.newaxis, :
            ]
        else:
            position_ids = tf.cast(position_ids, tf.keras.backend.floatx())

        # indices: 0..(output_dim//2 - 1)
        indices = tf.range(0, self.output_dim // 2, dtype=tf.keras.backend.floatx())
        # 10000^(-2i/d_model)
        indices = tf.pow(10000.0, -2.0 * indices / self.output_dim)

        # 形状 (B?, N?, D?)：这里遵循原公式：pos_embeddings[b, n, d] = pos[n] * indices[d]
        # 原实现使用了 tf.einsum('bn,d->bnd', ...)
        pos_embeddings = tf.einsum('bn,d->bnd', position_ids, indices)

        # sin/cos：先在最后一维加 singleton，再 reshape 回 (B, T, D)
        pos_embeddings = tf.concat(
            [tf.sin(pos_embeddings)[..., tf.newaxis], tf.cos(pos_embeddings)[..., tf.newaxis]],
            axis=-1,
        )
        pos_embeddings = tf.reshape(pos_embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        elif self.merge_mode == 'mul':
            return inputs * pos_embeddings
        else:
            if not self.custom_position_ids:
                pos_embeddings = tf.tile(pos_embeddings, [batch_size, 1, 1])
            return tf.concat([inputs, pos_embeddings], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]
        if self.merge_mode in ['add', 'mul']:
            return input_shape
        else:
            return (input_shape[0], input_shape[1], input_shape[2] + self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base.items()) + list(config.items()))
