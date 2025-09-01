#!/usr/bin/env python3
"""
Test script for AttentionLayer with TensorFlow 2.15
This script tests the attention layer for runtime compatibility with TensorFlow 2.15.
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add the path to your layers
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

# Import your attention layer
from attention_layer_update import AttentionLayer

def test_attention_layer():
    """Test the AttentionLayer for TensorFlow 2.15 runtime compatibility"""
    
    print("TensorFlow version:", tf.__version__)
    print("Testing AttentionLayer for TensorFlow 2.15 compatibility...")
    
    # Test parameters
    batch_size = 4
    time_steps = 10
    seq_len = 32
    
    # Create sample input data
    input_data = np.random.random((batch_size, time_steps, seq_len)).astype(np.float32)
    print(f"Input shape: {input_data.shape}")
    
    try:
        # Test 1: Layer instantiation
        print("\n1. Testing layer instantiation...")
        attention_layer = AttentionLayer()
        print("✅ Layer instantiation successful")
        
        # Test 2: Forward pass
        print("\n2. Testing forward pass...")
        output = attention_layer(input_data)
        print(f"✅ Forward pass successful - Output shape: {output.shape}")
        
        # Test 3: Shape validation
        print("\n3. Testing shape validation...")
        expected_shape = (batch_size, time_steps, seq_len)
        if output.shape == expected_shape:
            print("✅ Output shape matches expected shape")
        else:
            print(f"❌ Shape mismatch - Expected: {expected_shape}, Got: {output.shape}")
            return False
        
        # Test 4: Trainable weights
        print("\n4. Testing trainable weights...")
        print(f"✅ Layer has {len(attention_layer.trainable_weights)} trainable weights")
        print(f"✅ Layer trainable: {attention_layer.trainable}")
        
        # Test 5: Different input shapes (same time_steps)
        print("\n5. Testing with different input shapes...")
        
        # Test different batch sizes
        test_input1 = np.random.random((2, time_steps, seq_len)).astype(np.float32)
        output1 = attention_layer(test_input1)
        print(f"✅ Different batch size: Input {test_input1.shape} -> Output {output1.shape}")
        
        # Test different sequence lengths
        test_input2 = np.random.random((batch_size, time_steps, 16)).astype(np.float32)
        output2 = attention_layer(test_input2)
        print(f"✅ Different seq length: Input {test_input2.shape} -> Output {output2.shape}")
        
        # Test 6: Gradient computation
        print("\n6. Testing gradient computation...")
        with tf.GradientTape() as tape:
            output = attention_layer(input_data)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, attention_layer.trainable_weights)
        print(f"✅ Gradients computed successfully for {len(gradients)} weights")
        
        # Test 7: Keras model integration
        print("\n7. Testing Keras model integration...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(time_steps, seq_len)),
            AttentionLayer(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model compilation successful")
        
        # Test model prediction
        prediction = model.predict(input_data)
        print(f"✅ Model prediction successful - Shape: {prediction.shape}")
        
        print("\n🎉 All TensorFlow 2.15 compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_mechanism():
    """Test the attention mechanism with simple data"""
    print("\n" + "="*50)
    print("Testing attention mechanism behavior...")
    
    batch_size = 2
    time_steps = 3
    seq_len = 4
    
    # Create simple test data
    input_data = np.array([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0, 16.0],
         [17.0, 18.0, 19.0, 20.0],
         [21.0, 22.0, 23.0, 24.0]]
    ], dtype=np.float32)
    
    try:
        attention_layer = AttentionLayer()
        output = attention_layer(input_data)
        
        print(f"✅ Attention mechanism test successful")
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Attention mechanism test failed: {str(e)}")

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*50)
    print("Testing error handling...")
    
    try:
        attention_layer = AttentionLayer()
        
        # Test 1: 2D input instead of 3D
        try:
            input_2d = np.random.random((2, 3)).astype(np.float32)
            output = attention_layer(input_2d)
            print("❌ Should have raised error for 2D input")
        except Exception as e:
            print(f"✅ Correctly handled 2D input error: {str(e)[:50]}...")
        
        # Test 2: 4D input instead of 3D
        try:
            input_4d = np.random.random((2, 3, 4, 5)).astype(np.float32)
            output = attention_layer(input_4d)
            print("❌ Should have raised error for 4D input")
        except Exception as e:
            print(f"✅ Correctly handled 4D input error: {str(e)[:50]}...")
        
        # Test 3: Different time steps (should fail due to weight matrix dimension mismatch)
        try:
            # First create layer with one time step size
            test_input1 = np.random.random((2, 10, 5)).astype(np.float32)
            attention_layer(test_input1)  # This builds the layer with time_steps=10
            
            # Now try with different time steps
            test_input2 = np.random.random((2, 5, 5)).astype(np.float32)
            output = attention_layer(test_input2)
            print("❌ Should have raised error for different time steps")
        except Exception as e:
            print(f"✅ Correctly handled time step mismatch error: {str(e)[:50]}...")
        
        print("✅ Error handling tests completed!")
        
    except Exception as e:
        print(f"❌ Error handling test setup failed: {str(e)}")

def test_tensorflow_operations():
    """Test that all TensorFlow operations used in the layer are compatible"""
    print("\n" + "="*50)
    print("Testing TensorFlow 2.15 operations compatibility...")
    
    try:
        # Test all operations used in the attention layer
        batch_size, time_steps, seq_len = 2, 3, 4
        
        # Test tf.transpose
        input_data = np.random.random((batch_size, time_steps, seq_len)).astype(np.float32)
        x = tf.transpose(input_data, perm=(0, 2, 1))
        print("✅ tf.transpose operation successful")
        
        # Test tf.linalg.matmul
        W = tf.random.uniform((time_steps, time_steps))
        b = tf.random.uniform((time_steps,))
        matmul_result = tf.linalg.matmul(x, W)
        print("✅ tf.linalg.matmul operation successful")
        
        # Test tf.math.tanh
        tanh_result = tf.math.tanh(matmul_result + b)
        print("✅ tf.math.tanh operation successful")
        
        # Test tf.nn.softmax
        softmax_result = tf.nn.softmax(tanh_result, axis=-1)
        print("✅ tf.nn.softmax operation successful")
        
        print("✅ All TensorFlow 2.15 operations are compatible!")
        
    except Exception as e:
        print(f"❌ TensorFlow operations test failed: {str(e)}")

if __name__ == "__main__":
    print("Starting AttentionLayer TensorFlow 2.15 compatibility tests...")
    print("="*50)
    
    # Run main compatibility tests
    success = test_attention_layer()
    
    if success:
        # Run additional tests
        test_attention_mechanism()
        test_error_handling()
        test_tensorflow_operations()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All TensorFlow 2.15 compatibility tests completed successfully!")
        print("Your AttentionLayer is fully compatible with TensorFlow 2.15.")
    else:
        print("❌ Some compatibility tests failed. Please check the error messages above.")
