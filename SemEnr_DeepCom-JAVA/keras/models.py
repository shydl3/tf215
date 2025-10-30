# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
import logging
import numpy as np
import tensorflow as tf

# Keras (TF2.16) imports
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Concatenate, Dot, Embedding, Dropout, Lambda, Activation,
    LSTM, Dense, Reshape, Conv1D, Conv2D, SeparableConv2D, MaxPooling1D,
    Flatten, GlobalMaxPooling1D, Bidirectional, SimpleRNN,
    GlobalAveragePooling1D, GlobalAveragePooling2D, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Project layers
from layers.coattention_layer import COAttentionLayer
from layers.joint_self_attention_layer import JointSelfAttentionLayer
from layers.attention_layer import AttentionLayer
from layers.position_encoding_layer import PositionEncodingLayer, PositionEmbedding, SinusoidalPositionEmbedding
from layers.mul_dim_layer import MulDimLayer
from layers.layer_norm_layer import LayerNormLayer

logger = logging.getLogger(__name__)


class JointEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params', dict())

        # Inputs (training graph)
        self.tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='i_tokens')
        self.sim_desc = Input(shape=(self.data_params['sim_desc_len'],), dtype='int32', name='i_sim_desc')
        self.desc_good = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')
        self.desc_bad = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_bad')

        # Placeholders for built models
        self._sim_model = None
        self._training_model = None
        self._shared_model = None

        # Ensure model dir
        model_dir = os.path.join(self.config['workdir'], 'models', self.model_params['model_name'])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    def build(self):
        """
        1) Build Code Representation Subgraph
        """
        logger.debug('Building Code Representation Model')
        tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='tokens')
        sim_desc = Input(shape=(self.data_params['sim_desc_len'],), dtype='int32', name='sim_desc')

        # ---- Tokens Representation ----
        init_emb_weights = None
        if self.model_params.get('init_embed_weights_tokens') is not None:
            init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_tokens'])
            init_emb_weights = [init_emb_weights]
        embedding_tokens = Embedding(
            input_dim=self.data_params['n_tokens_words'],
            output_dim=self.model_params.get('n_embed_dims'),
            weights=init_emb_weights,
            trainable=True,
            mask_zero=False,
            name='embedding_tokens'
        )
        tokens_embedding = embedding_tokens(tokens)
        tokens_dropout = Dropout(0.25, name='dropout_tokens_embed')(tokens_embedding)
        tokens_out = AttentionLayer(name='tokens_attention_layer')(tokens_dropout)

        # ---- sim_desc Representation ----
        init_emb_weights = None
        if self.model_params.get('init_embed_weights_desc') is not None:
            init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_desc'])
            init_emb_weights = [init_emb_weights]
        embedding_sim_desc = Embedding(
            input_dim=self.data_params['n_desc_words'],
            output_dim=self.model_params.get('n_embed_dims'),
            weights=init_emb_weights,
            trainable=True,
            mask_zero=False,
            name='embedding_sim_desc'
        )
        sim_desc_embedding = embedding_sim_desc(sim_desc)
        sim_desc_dropout = Dropout(0.25, name='dropout_sim_desc_embed')(sim_desc_embedding)
        sim_desc_out = AttentionLayer(name='sim_desc_attention_layer')(sim_desc_dropout)

        """
        2) Build Desc Representation Subgraph
        """
        logger.debug('Building Desc Representation Model')
        desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        init_emb_weights = None
        if self.model_params.get('init_embed_weights_desc') is not None:
            init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_desc'])
            init_emb_weights = [init_emb_weights]
        embedding_desc = Embedding(
            input_dim=self.data_params['n_desc_words'],
            output_dim=self.model_params.get('n_embed_dims'),
            weights=init_emb_weights,
            trainable=True,
            mask_zero=False,
            name='embedding_desc'
        )
        desc_embedding = embedding_desc(desc)
        desc_dropout = Dropout(0.25, name='dropout_desc_embed')(desc_embedding)
        merged_desc = AttentionLayer(name='desc_attention_layer')(desc_dropout)

        # ---- AP networks (co-attention) ----
        attention = COAttentionLayer(name='coattention_layer')
        attention_tq_out = attention([tokens_out, merged_desc])
        attention_sq_out = attention([sim_desc_out, merged_desc])

        # Helpers / ops (TF2 replacements where needed)
        # tf.matrix_diag -> tf.linalg.diag in TF2
        normalOp = Lambda(lambda x: tf.linalg.diag(x), name='normalOp')
        gap_cnn = GlobalAveragePooling1D(name='globalaveragepool_cnn')
        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='trans_coattention')

        # ---- Branch tq (column-wise then row-wise) ----
        activ_tq_1 = Activation('softmax', name='tq_AP_active_colum')
        dot_tq_1 = Dot(axes=1, normalize=False, name='tq_column_dot')
        tq_conv1 = Conv1D(100, 2, padding='same', activation='relu', strides=1, name='tq_conv1')
        dense_tq_desc = Dense(30, use_bias=False, name='dense_tq_desc')

        attention_tq_matrix = attention_tq_out
        tq_desc_conv = tq_conv1(attention_tq_matrix)
        tq_desc_conv = dense_tq_desc(tq_desc_conv)
        tq_desc_conv = gap_cnn(tq_desc_conv)
        tq_desc_att = activ_tq_1(tq_desc_conv)
        tq_desc_out = dot_tq_1([tq_desc_att, merged_desc])

        attention_transposed = attention_trans_layer(attention_tq_out)
        activ_tq_2 = Activation('softmax', name='tq_AP_active_row')
        dot_tq_2 = Dot(axes=1, normalize=False, name='tq_row_dot')
        tq_conv2 = Conv1D(100, 2, padding='same', activation='relu', strides=1, name='tq_conv2')
        dense_tq = Dense(50, use_bias=False, name='dense_tq')

        attention_tq_matrix = attention_transposed
        tq_out_conv = tq_conv2(attention_tq_matrix)
        tq_out_conv = dense_tq(tq_out_conv)
        tq_out_conv = gap_cnn(tq_out_conv)
        tq_out_att = activ_tq_2(tq_out_conv)
        tq_out = dot_tq_2([tq_out_att, tokens_out])

        # ---- Branch sq (column-wise then row-wise) ----
        activ_sq_1 = Activation('softmax', name='sq_AP_active_colum')
        dot_sq_1 = Dot(axes=1, normalize=False, name='sq_column_dot')
        sq_conv1 = Conv1D(100, 2, padding='same', activation='relu', strides=1, name='sq_conv1')
        dense_sq_desc = Dense(30, use_bias=False, name='dense_sq_desc')

        attention_sq_matrix = attention_sq_out
        sq_desc_conv = sq_conv1(attention_sq_matrix)
        sq_desc_conv = dense_sq_desc(sq_desc_conv)
        sq_desc_conv = gap_cnn(sq_desc_conv)
        sq_desc_att = activ_sq_1(sq_desc_conv)
        sq_desc_out = dot_sq_1([sq_desc_att, merged_desc])

        attention_transposed = attention_trans_layer(attention_sq_out)
        activ_sq_2 = Activation('softmax', name='sq_AP_active_row')
        dot_sq_2 = Dot(axes=1, normalize=False, name='sq_row_dot')
        sq_conv2 = Conv1D(100, 2, padding='same', activation='relu', strides=1, name='sq_conv2')
        dense_sq = Dense(30, use_bias=False, name='dense_sq')

        attention_sq_matrix = attention_transposed
        sq_out_conv = sq_conv2(attention_sq_matrix)
        sq_out_conv = dense_sq(sq_out_conv)
        sq_out_conv = gap_cnn(sq_out_conv)
        sq_out_att = activ_sq_2(sq_out_conv)
        sq_out = dot_sq_2([sq_out_att, sim_desc_out])

        # ---- Merge desc/code and attend ----
        merged_desc_out = Concatenate(name='desc_orig_merge', axis=1)([tq_desc_out, sq_desc_out])
        merged_code_out = Concatenate(name='code_orig_merge', axis=1)([tq_out, sq_out])
        reshape_desc = Reshape((2, 100))(merged_desc_out)
        reshape_code = Reshape((2, 100))(merged_code_out)

        att_desc_out = AttentionLayer(name='desc_merged_attention_layer')(reshape_desc)
        att_code_out = AttentionLayer(name='code_merged_attention_layer')(reshape_code)

        gap = GlobalAveragePooling1D(name='blobalaveragepool')
        mulop = Lambda(lambda x: x * 2.0, name='mulop')
        desc_out = mulop(gap(att_desc_out))
        code_out = mulop(gap(att_code_out))

        """
        3) Similarity head (cosine sim)
        """
        logger.debug('Building similarity model')
        cos_sim = Dot(axes=1, normalize=True, name='cos_sim')([code_out, desc_out])

        sim_model = Model(inputs=[tokens, sim_desc, desc], outputs=[cos_sim], name='sim_model')
        self._sim_model = sim_model
        print("\nsummary of similarity model")
        self._sim_model.summary()
        # fname = os.path.join(self.config['workdir'], 'models', self.model_params['model_name'], '_sim_model.png')
        # plot_model(self._sim_model, show_shapes=True, to_file=fname)

        """
        4) Training graph with margin-based loss
        """
        good_sim = sim_model([self.tokens, self.sim_desc, self.desc_good])
        bad_sim = sim_model([self.tokens, self.sim_desc, self.desc_bad])
        margin = float(self.model_params.get('margin', 0.05))
        loss = Lambda(lambda x: K.maximum(1e-6, margin - x[0] + x[1]), name='loss')([good_sim, bad_sim])

        logger.debug('Building training model')
        self._training_model = Model(
            inputs=[self.tokens, self.sim_desc, self.desc_good, self.desc_bad],
            outputs=[loss],
            name='training_model'
        )
        print('\nsummary of training model')
        self._training_model.summary()
        # fname = os.path.join(self.config['workdir'], 'models', self.model_params['model_name'], '_training_model.png')
        # plot_model(self._training_model, show_shapes=True, to_file=fname)

    def compile(self, optimizer, **kwargs):
        logger.info('Compiling models (TF2/Keras)')
        # y_true is a dummy zero vector; ensure it's "used" to avoid warnings
        self._training_model.compile(
            loss=lambda y_true, y_pred: y_pred + y_true - y_true,
            optimizer=optimizer,
            **kwargs
        )
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)

    def save(self, sim_model_file, **kwargs):
        assert self._sim_model is not None, 'Must compile the model before saving weights'
        self._sim_model.save_weights(sim_model_file, **kwargs)

    def load(self, sim_model_file, **kwargs):
        assert self._sim_model is not None, 'Must compile the model before loading weights'
        self._sim_model.load_weights(sim_model_file, **kwargs)
