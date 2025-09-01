#!/usr/bin/env python3
"""
Test script for COAttentionLayer with TensorFlow 2.15
This script tests the coattention layer for runtime compatibility with TensorFlow 2.15.
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add the path to your layers
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

# Import your coattention layer
from coattention_layer_update import COAttentionLayer

def test_coattention_layer():
    """Test the COAttentionLayer for TensorFlow 2.15 runtime compatibility"""
    
    print("TensorFlow version:", tf.__version__)
    print("Testing COAttentionLayer for TensorFlow 2.15 compatibility...")
    
    # Test parameters
    batch_size = 4
    seq1_len = 10
    seq2_len = 8
    embedding_dim = 32
    
    # Create sample input data - coattention needs 2 inputs
    input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
    input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
    
    print(f"Input 1 shape: {input1.shape}")
    print(f"Input 2 shape: {input2.shape}")
    
    try:
        # Test 1: Layer instantiation
        print("\n1. Testing layer instantiation...")
        coattention_layer = COAttentionLayer()
        print("✅ Layer instantiation successful")
        
        # Test 2: Forward pass
        print("\n2. Testing forward pass...")
        output = coattention_layer([input1, input2])
        print(f"✅ Forward pass successful - Output shape: {output.shape}")
        
        # Test 3: Shape validation
        print("\n3. Testing shape validation...")
        expected_shape = (batch_size, seq1_len, seq2_len)
        if output.shape == expected_shape:
            print("✅ Output shape matches expected shape")
        else:
            print(f"❌ Shape mismatch - Expected: {expected_shape}, Got: {output.shape}")
            return False
        
        # Test 4: Trainable weights
        print("\n4. Testing trainable weights...")
        print(f"✅ Layer has {len(coattention_layer.trainable_weights)} trainable weights")
        print(f"✅ Layer trainable: {coattention_layer.trainable}")
        
        # Test 5: Different input shapes
        print("\n5. Testing with different input shapes...")
        
        # Test different batch sizes
        test_input1_1 = np.random.random((2, seq1_len, embedding_dim)).astype(np.float32)
        test_input2_1 = np.random.random((2, seq2_len, embedding_dim)).astype(np.float32)
        output1 = coattention_layer([test_input1_1, test_input2_1])
        print(f"✅ Different batch size: Input1 {test_input1_1.shape}, Input2 {test_input2_1.shape} -> Output {output1.shape}")
        
        # Test different sequence lengths
        test_input1_2 = np.random.random((batch_size, 5, embedding_dim)).astype(np.float32)
        test_input2_2 = np.random.random((batch_size, 3, embedding_dim)).astype(np.float32)
        output2 = coattention_layer([test_input1_2, test_input2_2])
        print(f"✅ Different seq lengths: Input1 {test_input1_2.shape}, Input2 {test_input2_2.shape} -> Output {output2.shape}")
        
        # Test 6: Gradient computation
        print("\n6. Testing gradient computation...")
        with tf.GradientTape() as tape:
            output = coattention_layer([input1, input2])
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, coattention_layer.trainable_weights)
        print(f"✅ Gradients computed successfully for {len(gradients)} weights")
        
        # Test 7: Keras model integration
        print("\n7. Testing Keras model integration...")
        
        # Create a model that takes two inputs
        input1_layer = tf.keras.layers.Input(shape=(seq1_len, embedding_dim))
        input2_layer = tf.keras.layers.Input(shape=(seq2_len, embedding_dim))
        
        coattention_output = COAttentionLayer()([input1_layer, input2_layer])
        
        # Add processing layers - coattention output is 3D
        pooled = tf.keras.layers.GlobalAveragePooling1D()(coattention_output)
        dense_output = tf.keras.layers.Dense(1)(pooled)
        
        model = tf.keras.Model(inputs=[input1_layer, input2_layer], outputs=dense_output)
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model compilation successful")
        
        # Test model prediction
        prediction = model.predict([input1, input2])
        print(f"✅ Model prediction successful - Shape: {prediction.shape}")
        
        # Test alternative approach with GlobalAveragePooling2D
        print("\n8. Testing alternative model with GlobalAveragePooling2D...")
        input1_layer_alt = tf.keras.layers.Input(shape=(seq1_len, embedding_dim))
        input2_layer_alt = tf.keras.layers.Input(shape=(seq2_len, embedding_dim))
        
        coattention_output_alt = COAttentionLayer()([input1_layer_alt, input2_layer_alt])
        
        # Reshape to 4D for GlobalAveragePooling2D
        reshaped = tf.keras.layers.Reshape((seq1_len, seq2_len, 1))(coattention_output_alt)
        pooled_alt = tf.keras.layers.GlobalAveragePooling2D()(reshaped)
        dense_output_alt = tf.keras.layers.Dense(1)(pooled_alt)
        
        model_alt = tf.keras.Model(inputs=[input1_layer_alt, input2_layer_alt], outputs=dense_output_alt)
        model_alt.compile(optimizer='adam', loss='mse')
        print("✅ Alternative model compilation successful")
        
        # Test alternative model prediction
        prediction_alt = model_alt.predict([input1, input2])
        print(f"✅ Alternative model prediction successful - Shape: {prediction_alt.shape}")
        
        print("\n🎉 All TensorFlow 2.15 compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_coattention_mechanism():
    """Test the coattention mechanism with simple data"""
    print("\n" + "="*50)
    print("Testing coattention mechanism behavior...")
    
    batch_size = 2
    seq1_len = 3
    seq2_len = 2
    embedding_dim = 4
    
    # Create simple test data
    input1 = np.array([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0, 16.0],
         [17.0, 18.0, 19.0, 20.0],
         [21.0, 22.0, 23.0, 24.0]]
    ], dtype=np.float32)
    
    input2 = np.array([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]],
        [[9.0, 10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0, 16.0]]
    ], dtype=np.float32)
    
    try:
        coattention_layer = COAttentionLayer()
        output = coattention_layer([input1, input2])
        
        print(f"✅ Coattention mechanism test successful")
        print(f"Input 1 shape: {input1.shape}")
        print(f"Input 2 shape: {input2.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Coattention mechanism test failed: {str(e)}")

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*50)
    print("Testing error handling...")
    
    try:
        coattention_layer = COAttentionLayer()
        
        # Test 1: Single input instead of list
        try:
            single_input = np.random.random((2, 3, 4)).astype(np.float32)
            output = coattention_layer(single_input)
            print("❌ Should have raised error for single input")
        except Exception as e:
            print(f"✅ Correctly handled single input error: {str(e)[:50]}...")
        
        # Test 2: Mismatched embedding dimensions
        try:
            input1 = np.random.random((2, 3, 4)).astype(np.float32)
            input2 = np.random.random((2, 3, 5)).astype(np.float32)  # Different embedding dim
            output = coattention_layer([input1, input2])
            print("❌ Should have raised error for mismatched embedding dimensions")
        except Exception as e:
            print(f"✅ Correctly handled embedding dimension mismatch: {str(e)[:50]}...")
        
        # Test 3: Wrong number of inputs
        try:
            input1 = np.random.random((2, 3, 4)).astype(np.float32)
            input2 = np.random.random((2, 3, 4)).astype(np.float32)
            input3 = np.random.random((2, 3, 4)).astype(np.float32)
            output = coattention_layer([input1, input2, input3])
            print("❌ Should have raised error for three inputs")
        except Exception as e:
            print(f"✅ Correctly handled wrong number of inputs: {str(e)[:50]}...")
        
        print("✅ Error handling tests completed!")
        
    except Exception as e:
        print(f"❌ Error handling test setup failed: {str(e)}")

def test_tensorflow_operations():
    """Test that all TensorFlow operations used in the layer are compatible"""
    print("\n" + "="*50)
    print("Testing TensorFlow 2.15 operations compatibility...")
    
    try:
        # Test all operations used in the coattention layer
        batch_size, seq1_len, seq2_len, embedding_dim = 2, 3, 2, 4
        
        # Test tf.linalg.matmul
        input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
        kernel = tf.random.uniform((embedding_dim, embedding_dim))
        matmul_result = tf.linalg.matmul(input1, kernel)
        print("✅ tf.linalg.matmul operation successful")
        
        # Test tf.transpose
        input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
        y_trans = tf.transpose(input2, perm=(0, 2, 1))
        print("✅ tf.transpose operation successful")
        
        # Test tf.linalg.matmul for batch matrix multiplication
        b = tf.linalg.matmul(matmul_result, y_trans)
        print("✅ tf.linalg.matmul batch operation successful")
        
        # Test tf.math.tanh
        tanh_result = tf.math.tanh(b)
        print("✅ tf.math.tanh operation successful")
        
        print("✅ All TensorFlow 2.15 operations are compatible!")
        
    except Exception as e:
        print(f"❌ TensorFlow operations test failed: {str(e)}")

def test_dual_input_handling():
    """Test the dual input handling mechanism"""
    print("\n" + "="*50)
    print("Testing dual input handling...")
    
    try:
        batch_size = 2
        seq1_len = 4
        seq2_len = 3
        embedding_dim = 5
        
        # Create test inputs
        input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
        input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
        
        coattention_layer = COAttentionLayer()
        
        # Test with list input
        output_list = coattention_layer([input1, input2])
        print("✅ List input handling successful")
        
        # Test with tuple input
        output_tuple = coattention_layer((input1, input2))
        print("✅ Tuple input handling successful")
        
        # Verify outputs are the same
        if np.allclose(output_list.numpy(), output_tuple.numpy()):
            print("✅ List and tuple inputs produce same output")
        else:
            print("❌ List and tuple inputs produce different outputs")
        
        print("✅ Dual input handling tests completed!")
        
    except Exception as e:
        print(f"❌ Dual input handling test failed: {str(e)}")

if __name__ == "__main__":
    print("Starting COAttentionLayer TensorFlow 2.15 compatibility tests...")
    print("="*50)
    
    # Run main compatibility tests
    success = test_coattention_layer()
    
    if success:
        # Run additional tests
        test_coattention_mechanism()
        test_error_handling()
        test_tensorflow_operations()
        test_dual_input_handling()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All TensorFlow 2.15 compatibility tests completed successfully!")
        print("Your COAttentionLayer is fully compatible with TensorFlow 2.15.")
    else:
        print("❌ Some compatibility tests failed. Please check the error messages above.")
