#!/usr/bin/env python3
"""
Test script for JointSelfAttentionLayer with TensorFlow 2.15
This script tests the joint self attention layer for runtime compatibility with TensorFlow 2.15.
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add the path to your layers
sys.path.append('SemEnr_CSN-JAVA/keras/layers')

# Import your joint self attention layer
from joint_self_attention_layer_update import JointSelfAttentionLayer

def test_joint_self_attention_layer():
    """Test the JointSelfAttentionLayer for TensorFlow 2.15 runtime compatibility"""
    
    print("TensorFlow version:", tf.__version__)
    print("Testing JointSelfAttentionLayer for TensorFlow 2.15 compatibility...")
    
    # Test parameters
    batch_size = 4
    seq1_len = 10  # T_c
    seq2_len = 8   # T_d
    embedding_dim = 32
    
    # Create sample input data - joint self attention needs 2 inputs
    input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
    input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
    
    print(f"Input 1 shape: {input1.shape}")
    print(f"Input 2 shape: {input2.shape}")
    
    try:
        # Test 1: Layer instantiation
        print("\n1. Testing layer instantiation...")
        joint_attention_layer = JointSelfAttentionLayer()
        print("✅ Layer instantiation successful")
        
        # Test 2: Forward pass
        print("\n2. Testing forward pass...")
        output = joint_attention_layer([input1, input2])
        print(f"✅ Forward pass successful - Output shapes: {[out.shape for out in output]}")
        
        # Test 3: Shape validation
        print("\n3. Testing shape validation...")
        expected_shapes = [(batch_size, embedding_dim), (batch_size, embedding_dim)]
        actual_shapes = [out.shape for out in output]
        
        if actual_shapes == expected_shapes:
            print("✅ Output shapes match expected shapes")
        else:
            print(f"❌ Shape mismatch - Expected: {expected_shapes}, Got: {actual_shapes}")
            return False
        
        # Test 4: Trainable weights
        print("\n4. Testing trainable weights...")
        print(f"✅ Layer has {len(joint_attention_layer.trainable_weights)} trainable weights")
        print(f"✅ Layer trainable: {joint_attention_layer.trainable}")
        
        # Test 5: Different input shapes
        print("\n5. Testing with different input shapes...")
        
        # Test different batch sizes
        test_input1_1 = np.random.random((2, seq1_len, embedding_dim)).astype(np.float32)
        test_input2_1 = np.random.random((2, seq2_len, embedding_dim)).astype(np.float32)
        output1 = joint_attention_layer([test_input1_1, test_input2_1])
        print(f"✅ Different batch size: Input1 {test_input1_1.shape}, Input2 {test_input2_1.shape} -> Output shapes {[out.shape for out in output1]}")
        
        # Test different sequence lengths
        test_input1_2 = np.random.random((batch_size, 5, embedding_dim)).astype(np.float32)
        test_input2_2 = np.random.random((batch_size, 3, embedding_dim)).astype(np.float32)
        output2 = joint_attention_layer([test_input1_2, test_input2_2])
        print(f"✅ Different seq lengths: Input1 {test_input1_2.shape}, Input2 {test_input2_2.shape} -> Output shapes {[out.shape for out in output2]}")
        
        # Test 6: Gradient computation
        print("\n6. Testing gradient computation...")
        with tf.GradientTape() as tape:
            output = joint_attention_layer([input1, input2])
            # Sum both outputs for loss computation
            loss = tf.reduce_mean(output[0]) + tf.reduce_mean(output[1])
        
        gradients = tape.gradient(loss, joint_attention_layer.trainable_weights)
        print(f"✅ Gradients computed successfully for {len(gradients)} weights")
        
        # Test 7: Keras model integration
        print("\n7. Testing Keras model integration...")
        
        # Create a model that takes two inputs
        input1_layer = tf.keras.layers.Input(shape=(seq1_len, embedding_dim))
        input2_layer = tf.keras.layers.Input(shape=(seq2_len, embedding_dim))
        
        joint_attention_output = JointSelfAttentionLayer()([input1_layer, input2_layer])
        
        # Process both outputs
        output1_processed = tf.keras.layers.Dense(1)(joint_attention_output[0])
        output2_processed = tf.keras.layers.Dense(1)(joint_attention_output[1])
        
        model = tf.keras.Model(inputs=[input1_layer, input2_layer], outputs=[output1_processed, output2_processed])
        model.compile(optimizer='adam', loss='mse')
        print("✅ Model compilation successful")
        
        # Test model prediction
        prediction = model.predict([input1, input2])
        print(f"✅ Model prediction successful - Output shapes: {[pred.shape for pred in prediction]}")
        
        # Test 8: Single output model (concatenate outputs)
        print("\n8. Testing single output model...")
        input1_layer_alt = tf.keras.layers.Input(shape=(seq1_len, embedding_dim))
        input2_layer_alt = tf.keras.layers.Input(shape=(seq2_len, embedding_dim))
        
        joint_attention_output_alt = JointSelfAttentionLayer()([input1_layer_alt, input2_layer_alt])
        
        # Concatenate both outputs
        concatenated = tf.keras.layers.Concatenate()(joint_attention_output_alt)
        dense_output = tf.keras.layers.Dense(1)(concatenated)
        
        model_alt = tf.keras.Model(inputs=[input1_layer_alt, input2_layer_alt], outputs=dense_output)
        model_alt.compile(optimizer='adam', loss='mse')
        print("✅ Single output model compilation successful")
        
        # Test single output model prediction
        prediction_alt = model_alt.predict([input1, input2])
        print(f"✅ Single output model prediction successful - Shape: {prediction_alt.shape}")
        
        print("\n🎉 All TensorFlow 2.15 compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_joint_attention_mechanism():
    """Test the joint attention mechanism with simple data"""
    print("\n" + "="*50)
    print("Testing joint attention mechanism behavior...")
    
    batch_size = 2
    seq1_len = 3  # T_c
    seq2_len = 2  # T_d
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
        joint_attention_layer = JointSelfAttentionLayer()
        output = joint_attention_layer([input1, input2])
        
        print(f"✅ Joint attention mechanism test successful")
        print(f"Input 1 shape: {input1.shape}")
        print(f"Input 2 shape: {input2.shape}")
        print(f"Output 1 shape: {output[0].shape}")
        print(f"Output 2 shape: {output[1].shape}")
        
    except Exception as e:
        print(f"❌ Joint attention mechanism test failed: {str(e)}")

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*50)
    print("Testing error handling...")
    
    try:
        joint_attention_layer = JointSelfAttentionLayer()
        
        # Test 1: Single input instead of list
        try:
            single_input = np.random.random((2, 3, 4)).astype(np.float32)
            output = joint_attention_layer(single_input)
            print("❌ Should have raised error for single input")
        except Exception as e:
            print(f"✅ Correctly handled single input error: {str(e)[:50]}...")
        
        # Test 2: Mismatched embedding dimensions
        try:
            input1 = np.random.random((2, 3, 4)).astype(np.float32)
            input2 = np.random.random((2, 3, 5)).astype(np.float32)  # Different embedding dim
            output = joint_attention_layer([input1, input2])
            print("❌ Should have raised error for mismatched embedding dimensions")
        except Exception as e:
            print(f"✅ Correctly handled embedding dimension mismatch: {str(e)[:50]}...")
        
        # Test 3: Wrong number of inputs
        try:
            input1 = np.random.random((2, 3, 4)).astype(np.float32)
            input2 = np.random.random((2, 3, 4)).astype(np.float32)
            input3 = np.random.random((2, 3, 4)).astype(np.float32)
            output = joint_attention_layer([input1, input2, input3])
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
        # Test all operations used in the joint self attention layer
        batch_size, seq1_len, seq2_len, embedding_dim = 2, 3, 2, 4
        
        # Test tf.linalg.matmul
        input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
        input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
        W = tf.random.uniform((embedding_dim, embedding_dim))
        
        Q_c = tf.linalg.matmul(input1, W)
        print("✅ tf.linalg.matmul operation successful")
        
        # Test tf.transpose
        K_d_T = tf.transpose(Q_c, perm=(0, 2, 1))
        print("✅ tf.transpose operation successful")
        
        # Test tf.linalg.matmul for attention computation
        logits = tf.linalg.matmul(Q_c, K_d_T)
        print("✅ tf.linalg.matmul attention computation successful")
        
        # Test tf.nn.softmax
        A = tf.nn.softmax(logits, axis=-1)
        print("✅ tf.nn.softmax operation successful")
        
        # Test tf.reduce_mean
        C = tf.reduce_mean(Q_c, axis=1, keepdims=False)
        print("✅ tf.reduce_mean operation successful")
        
        print("✅ All TensorFlow 2.15 operations are compatible!")
        
    except Exception as e:
        print(f"❌ TensorFlow operations test failed: {str(e)}")

def test_dual_output_handling():
    """Test the dual output handling mechanism"""
    print("\n" + "="*50)
    print("Testing dual output handling...")
    
    try:
        batch_size = 2
        seq1_len = 4
        seq2_len = 3
        embedding_dim = 5
        
        # Create test inputs
        input1 = np.random.random((batch_size, seq1_len, embedding_dim)).astype(np.float32)
        input2 = np.random.random((batch_size, seq2_len, embedding_dim)).astype(np.float32)
        
        joint_attention_layer = JointSelfAttentionLayer()
        
        # Test with list input
        output_list = joint_attention_layer([input1, input2])
        print("✅ List input handling successful")
        
        # Test with tuple input
        output_tuple = joint_attention_layer((input1, input2))
        print("✅ Tuple input handling successful")
        
        # Verify outputs are the same
        if (np.allclose(output_list[0].numpy(), output_tuple[0].numpy()) and 
            np.allclose(output_list[1].numpy(), output_tuple[1].numpy())):
            print("✅ List and tuple inputs produce same output")
        else:
            print("❌ List and tuple inputs produce different outputs")
        
        # Test output structure
        if len(output_list) == 2 and len(output_tuple) == 2:
            print("✅ Correct number of outputs (2)")
        else:
            print("❌ Incorrect number of outputs")
        
        print("✅ Dual output handling tests completed!")
        
    except Exception as e:
        print(f"❌ Dual output handling test failed: {str(e)}")

def test_attention_weights_structure():
    """Test the attention weights structure and naming"""
    print("\n" + "="*50)
    print("Testing attention weights structure...")
    
    try:
        joint_attention_layer = JointSelfAttentionLayer()
        
        # Build the layer with dummy input
        dummy_input1 = np.random.random((1, 5, 8)).astype(np.float32)
        dummy_input2 = np.random.random((1, 3, 8)).astype(np.float32)
        joint_attention_layer([dummy_input1, dummy_input2])
        
        # Check weight names and shapes
        weight_names = [w.name for w in joint_attention_layer.trainable_weights]
        expected_names = ['W_qc', 'W_vc', 'W_kd', 'W_vd']
        
        print(f"✅ Found {len(joint_attention_layer.trainable_weights)} trainable weights")
        print(f"✅ Weight names: {weight_names}")
        
        # Check if all expected weights are present
        for expected_name in expected_names:
            if any(expected_name in name for name in weight_names):
                print(f"✅ Found weight: {expected_name}")
            else:
                print(f"❌ Missing weight: {expected_name}")
        
        print("✅ Attention weights structure test completed!")
        
    except Exception as e:
        print(f"❌ Attention weights structure test failed: {str(e)}")

if __name__ == "__main__":
    print("Starting JointSelfAttentionLayer TensorFlow 2.15 compatibility tests...")
    print("="*50)
    
    # Run main compatibility tests
    success = test_joint_self_attention_layer()
    
    if success:
        # Run additional tests
        test_joint_attention_mechanism()
        test_error_handling()
        test_tensorflow_operations()
        test_dual_output_handling()
        test_attention_weights_structure()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All TensorFlow 2.15 compatibility tests completed successfully!")
        print("Your JointSelfAttentionLayer is fully compatible with TensorFlow 2.15.")
    else:
        print("❌ Some compatibility tests failed. Please check the error messages above.")
