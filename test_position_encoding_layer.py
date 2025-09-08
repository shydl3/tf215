
#!/usr/bin/env python3
"""
Tests for PositionEncodingLayer / PositionEmbedding / SinusoidalPositionEmbedding
(Compatible with TensorFlow 2.15)
Covers: forward shape/value sanity, model integration, tf.function, custom ids/merge modes.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# 按你的项目约定，加入 layers 路径（如有不同请自行调整）
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

from position_encoding_layer_update import (
    PositionEncodingLayer,
    PositionEmbedding,
    SinusoidalPositionEmbedding,
    positional_encoding,
)

def _print(ok, msg):
    print(("✅ " if ok else "❌ ") + msg)

def test_position_encoding_layer():
    print("\n=== Test: PositionEncodingLayer ===")
    B, T, D = 2, 7, 16
    x = np.random.randn(B, T, D).astype(np.float32)

    layer = PositionEncodingLayer()
    y = layer(x)
    _print(y.shape == (B, T, D), f"output shape == {(B, T, D)}")

    # 数值校验：y - x 应等于 positional_encoding(T, D)
    pe = positional_encoding(T, D).astype(np.float32)
    diff = np.max(np.abs(y.numpy() - (x + pe)))
    _print(diff < 1e-6, f"max |y - (x + PE)| < 1e-6 (got {diff:.3e})")

    # Keras model 集成
    inp = tf.keras.layers.Input(shape=(T, D))
    out = PositionEncodingLayer()(inp)
    model = tf.keras.Model(inp, out)
    pred = model.predict(x, verbose=0)
    _print(np.allclose(pred, x + pe, atol=1e-6), "Model predict matches x + PE")

def test_position_embedding_variants():
    print("\n=== Test: PositionEmbedding (add/mul/concat, hierarchical, custom ids) ===")
    B, T, D = 2, 6, 12
    x = np.random.randn(B, T, D).astype(np.float32)

    # 1) add
    pe_add = PositionEmbedding(input_dim=128, output_dim=D, merge_mode='add')
    y1 = pe_add(x)
    _print(y1.shape == (B, T, D), "add: output shape matches")

    # 2) mul
    pe_mul = PositionEmbedding(input_dim=128, output_dim=D, merge_mode='mul')
    y2 = pe_mul(x)
    _print(y2.shape == (B, T, D), "mul: output shape matches")

    # 3) concat
    pe_cat = PositionEmbedding(input_dim=128, output_dim=D, merge_mode='concat')
    y3 = pe_cat(x)
    _print(y3.shape == (B, T, D + D), "concat: output shape matches")

    # 4) hierarchical (alpha=True -> 0.4)
    pe_h = PositionEmbedding(input_dim=8, output_dim=D, merge_mode='add', hierarchical=True)
    y4 = pe_h(x)
    _print(y4.shape == (B, T, D), "hierarchical add: output shape matches")

    # 5) custom_position_ids
    pos_ids = np.tile(np.arange(T, dtype=np.int32)[None, :], (B, 1))
    pe_c = PositionEmbedding(input_dim=256, output_dim=D, merge_mode='add', custom_position_ids=True)
    y5 = pe_c([x, pos_ids])
    _print(y5.shape == (B, T, D), "custom ids(add): output shape matches")

def test_sinusoidal_position_embedding():
    print("\n=== Test: SinusoidalPositionEmbedding (add/mul/concat, custom ids) ===")
    B, T, D = 2, 5, 10
    x = np.random.randn(B, T, D).astype(np.float32)

    # add
    s_add = SinusoidalPositionEmbedding(output_dim=D, merge_mode='add')
    y1 = s_add(x)
    _print(y1.shape == (B, T, D), "sinusoid add: output shape matches")

    # mul
    s_mul = SinusoidalPositionEmbedding(output_dim=D, merge_mode='mul')
    y2 = s_mul(x)
    _print(y2.shape == (B, T, D), "sinusoid mul: output shape matches")

    # concat
    s_cat = SinusoidalPositionEmbedding(output_dim=D, merge_mode='concat')
    y3 = s_cat(x)
    _print(y3.shape == (B, T, D + D), "sinusoid concat: output shape matches")

    # custom ids
    pos_ids = np.tile(np.arange(T, dtype=np.int32)[None, :], (B, 1))
    s_c = SinusoidalPositionEmbedding(output_dim=D, merge_mode='add', custom_position_ids=True)
    y4 = s_c([x, pos_ids])
    _print(y4.shape == (B, T, D), "sinusoid custom ids(add): output shape matches")

def test_model_compile_and_tf_function():
    print("\n=== Test: Model compile & @tf.function ===")
    B, T, D = 2, 8, 16
    x = np.random.randn(B, T, D).astype(np.float32)
    y = np.random.randn(B, 1).astype(np.float32)

    inp = tf.keras.layers.Input(shape=(T, D))
    out = PositionEmbedding(input_dim=512, output_dim=D, merge_mode='add')(inp)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(out)
    head = tf.keras.layers.Dense(1)(pooled)
    model = tf.keras.Model(inp, head)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    _print(True, "Model compile/fit OK")

    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            preds = model(batch_x, training=True)
            loss = tf.reduce_mean(tf.square(preds - batch_y))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    loss_val = train_step(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
    _print(np.isfinite(loss_val.numpy()), f"@tf.function step OK (loss={loss_val.numpy():.6f})")

if __name__ == "__main__":
    print("TensorFlow:", tf.__version__)
    ok = True
    try:
        test_position_encoding_layer()
        test_position_embedding_variants()
        test_sinusoidal_position_embedding()
        test_model_compile_and_tf_function()
    except Exception as e:
        ok = False
        import traceback; traceback.print_exc()
        print("❌ Failed with error:", e)
    print("\nResult:", "PASS" if ok else "FAIL")
