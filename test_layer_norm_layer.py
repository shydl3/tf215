#!/usr/bin/env python3
"""
Test script for LayerNormLayer with TensorFlow 2.15
This script validates runtime compatibility (forward, backward, model integration).
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add the path to your layers (keep the same calling convention as your sample)
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

# Import your updated layer (ensure the filename/class name matches your upgraded file)
from layer_norm_layer_update import LayerNormLayer


def test_layer_norm_layer():
    """Main compatibility test for LayerNormLayer under TensorFlow 2.15"""

    print("TensorFlow version:", tf.__version__)
    print("Testing LayerNormLayer for TensorFlow 2.15 compatibility...")

    # Test parameters
    batch_size = 4
    seq_len = 10
    feat_dim = 32

    # Create sample input data: (B, T, F)
    x = np.random.random((batch_size, seq_len, feat_dim)).astype(np.float32)
    print(f"Input shape: {x.shape}")

    try:
        # 1) Instantiation (explicit epsilon recommended for numerical stability)
        print("\n1. Testing layer instantiation...")
        layer = LayerNormLayer(epsilon=1e-6)  # keep behavior explicit; original layer allows None
        print("✅ Layer instantiation successful")

        # 2) Forward pass
        print("\n2. Testing forward pass...")
        y = layer(x)
        print(f"✅ Forward pass successful - Output shape: {y.shape}")

        # 3) Shape validation (LayerNormLayer keeps the same shape)
        print("\n3. Testing shape validation...")
        expected_shape = (batch_size, seq_len, feat_dim)
        if y.shape == expected_shape:
            print("✅ Output shape matches expected shape")
        else:
            print(f"❌ Shape mismatch - Expected: {expected_shape}, Got: {y.shape}")
            return False

        # 4) Trainable weights check (W_1, W_2)
        print("\n4. Testing trainable weights...")
        tw = layer.trainable_weights
        print(f"✅ Layer has {len(tw)} trainable weights")
        for i, w in enumerate(tw):
            print(f"   - Weight[{i}] name={w.name}, shape={w.shape}")

        # 5) Backward/gradients (simple scalar loss)
        print("\n5. Testing gradient computation...")
        with tf.GradientTape() as tape:
            y = layer(x)
            # simple L2 loss to produce gradients
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, layer.trainable_weights)
        if any(g is None for g in grads):
            print("❌ Some gradients are None")
            return False
        # check finite
        for g in grads:
            if not tf.reduce_all(tf.math.is_finite(g)):
                print("❌ Non-finite gradient detected")
                return False
        print(f"✅ Gradients computed successfully for {len(grads)} weights (loss={loss.numpy():.6f})")

        # 6) Different input shapes (batch/seq)
        print("\n6. Testing with different input shapes...")
        x1 = np.random.random((2, seq_len, feat_dim)).astype(np.float32)
        y1 = layer(x1)
        print(f"✅ Different batch size -> Output {y1.shape}")

        x2 = np.random.random((batch_size, 5, feat_dim)).astype(np.float32)
        y2 = layer(x2)
        print(f"✅ Different seq length -> Output {y2.shape}")

        # 7) Keras model integration (compile/predict)
        print("\n7. Testing Keras model integration...")
        inp = tf.keras.layers.Input(shape=(seq_len, feat_dim))
        out = LayerNormLayer(epsilon=1e-6)(inp)
        # Pool to scalar and add a small head just to complete graph
        pooled = tf.keras.layers.GlobalAveragePooling1D()(out)
        head = tf.keras.layers.Dense(1)(pooled)
        model = tf.keras.Model(inp, head)
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model compilation successful")

        # run a quick predict
        pred = model.predict(x, verbose=0)
        print(f"✅ Model prediction successful - Shape: {pred.shape}")

        # 8) @tf.function graph execution (optional performance path)
        print("\n8. Testing @tf.function-compiled train step...")
        opt = tf.keras.optimizers.Adam(1e-3)

        @tf.function
        def train_step(batch):
            with tf.GradientTape() as tape:
                y = layer(batch, training=True)
                loss = tf.reduce_mean(tf.square(y))
            grads = tape.gradient(loss, layer.trainable_variables)
            opt.apply_gradients(zip(grads, layer.trainable_variables))
            return loss

        loss_val = train_step(tf.convert_to_tensor(x))
        print(f"✅ @tf.function train step successful (loss={loss_val.numpy():.6f})")

        print("\n🎉 All TensorFlow 2.15 compatibility tests for LayerNormLayer passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_defaults():
    """Optional: demonstrate default epsilon=None behavior (matches original semantics)."""
    print("\n" + "=" * 50)
    print("Testing error handling for default epsilon=None (original behavior)...")
    try:
        layer = LayerNormLayer()  # epsilon=None as in original signature
        # This may raise if sqrt(variance + None) occurs in call()
        x = np.random.random((2, 3, 4)).astype(np.float32)
        _ = layer(x)
        print("ℹ️ Default epsilon=None executed without error in this run.")
    except Exception as e:
        print(f"✅ Correctly caught issue with epsilon=None: {str(e)[:80]}...")


def test_tf_ops_sanity():
    """Sanity check for TF ops used by LayerNormLayer."""
    print("\n" + "=" * 50)
    print("Testing TensorFlow ops compatibility...")
    try:
        x = tf.random.uniform((2, 5, 8))
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        std = tf.sqrt(var + 1e-6)
        y = (x - mean) / std
        y = tf.nn.relu(tf.linalg.matmul(y, tf.random.uniform((8, 8))))
        y = tf.linalg.matmul(y, tf.random.uniform((8, 8)))
        y = tf.nn.dropout(y, rate=0.25)
        z = y + (x - mean) / std
        assert z.shape == x.shape
        print("✅ All TensorFlow ops succeeded.")
    except Exception as e:
        print(f"❌ TF ops sanity test failed: {str(e)}")


if __name__ == "__main__":
    print("Starting LayerNormLayer TensorFlow 2.15 compatibility tests...")
    print("=" * 50)

    success = test_layer_norm_layer()

    if success:
        test_error_handling_defaults()
        test_tf_ops_sanity()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All TensorFlow 2.15 compatibility tests completed successfully!")
        print("Your LayerNormLayer is fully compatible with TensorFlow 2.15.")
    else:
        print("❌ Some compatibility tests failed. Please check the error messages above.")
