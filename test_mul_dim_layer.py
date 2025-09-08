#!/usr/bin/env python3
"""
Runtime compatibility test for MulDimLayer on TensorFlow 2.15
Covers: instantiation, forward, gradients, model integration, tf.function.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# 让脚本与项目里的导入方式保持一致（按需修改路径）
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

from mul_dim_layer_update import MulDimLayer  # 确保文件名/类名与上面一致


def test_mul_dim_layer():
    print("TensorFlow:", tf.__version__)
    print("Testing MulDimLayer...")

    # 基本形状
    B, T, D = 4, 6, 32
    x = np.random.randn(B, T, D).astype(np.float32)

    try:
        # 1) 实例化 & 前向
        print("\n1) Instantiation & Forward")
        layer = MulDimLayer()
        y = layer(x)
        print("   Output shape:", y.shape)
        assert y.shape == (B, T, D)

        # 验证数值：应等于输入 * sqrt(D)
        scale = np.sqrt(D).astype(np.float32)
        diff = np.max(np.abs(y.numpy() - (x * scale)))
        print("   Max abs diff vs expected:", float(diff))
        assert diff < 1e-6

        # 2) 梯度检查（挂在一个小模型上）
        print("\n2) Gradient Check")
        with tf.GradientTape() as tape:
            tape.watch(layer.trainable_variables)  # 没有可训练参数，这里为空列表
            y2 = layer(x)
            loss = tf.reduce_mean(tf.square(y2))
        grads = tape.gradient(loss, layer.trainable_variables)
        # 该层无 trainable_variables，确保不会报错即可
        print("   Trainable vars:", len(layer.trainable_variables))
        assert grads == [] or grads is None
        print("   Gradient path OK (no trainable vars)")

        # 3) Keras 模型集成（compile/predict）
        print("\n3) Keras Model Integration")
        inp = tf.keras.layers.Input(shape=(T, D))
        out = MulDimLayer()(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer='adam', loss='mse')
        pred = model.predict(x, verbose=0)
        print("   Predict OK, shape:", pred.shape)
        assert pred.shape == (B, T, D)

        # 4) @tf.function 训练步
        print("\n4) @tf.function Train Step")
        opt = tf.keras.optimizers.Adam(1e-3)

        @tf.function
        def train_step(batch):
            with tf.GradientTape() as tape:
                out = layer(batch, training=True)
                loss = tf.reduce_mean(tf.square(out))
            # 该层无可训练参数，这里只是验证路径不报错
            grads = tape.gradient(loss, layer.trainable_variables)
            if grads:
                opt.apply_gradients(zip(grads, layer.trainable_variables))
            return loss

        loss_val = train_step(tf.convert_to_tensor(x))
        print(f"   Train step OK, loss={loss_val.numpy():.6f}")

        # 5) 保存/加载（SavedModel）
        print("\n5) Save/Load Roundtrip (SavedModel)")
        tmp_dir = "./_tmp_mul_dim_savedmodel"
        model.save(tmp_dir, overwrite=True)
        reloaded = tf.keras.models.load_model(tmp_dir)
        pred2 = reloaded.predict(x, verbose=0)
        assert np.allclose(pred, pred2, atol=1e-6)
        print("   Save/Load OK")

        print("\n🎉 MulDimLayer all checks passed!")
        return True

    except Exception as e:
        print("❌ Test failed:", str(e))
        import traceback; traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = test_mul_dim_layer()
    print("\nResult:", "PASS" if ok else "FAIL")
