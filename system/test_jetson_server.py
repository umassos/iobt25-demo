#!/usr/bin/env python3
"""
Test script for Jetson Single Server
This script sends a test request to verify the server is working correctly.
"""

import grpc
import numpy as np
import time
from inference_pb2 import PredictRequest
from inference_pb2_grpc import EncoderServiceStub
import sys
import os

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_server(host="localhost", port=8180, model_name="EENetB0_1_FC"):
    """Test the single server with a dummy request."""
    
    # Create gRPC channel and stub
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = EncoderServiceStub(channel)
    
    print(f"Testing server at {host}:{port}")
    print(f"Model: {model_name}")
    
    try:
        # Create a dummy input (adjust shape based on your model)
        # This is a placeholder - you'll need to adjust based on your actual model input shape
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)  # Example for image input
        
        # Create request
        request = PredictRequest(
            request_id=1,
            input=dummy_input.tobytes(),
            shape=list(dummy_input.shape)
        )
        
        print(f"Input shape: {dummy_input.shape}")
        print("Sending request...")
        
        # Send request and measure time
        start_time = time.time()
        response = stub.Predict(request)
        end_time = time.time()
        
        print(f"Response received in {end_time - start_time:.3f} seconds")
        print(f"Output shape: {response.shape}")
        print(f"Full model: {response.full_model}")
        print(f"Has result: {response.has_result}")
        
        # Convert output back to numpy array
        output_array = np.frombuffer(response.output, dtype=np.float32).reshape(response.shape)
        print(f"Output array shape: {output_array.shape}")
        print(f"Output min/max: {output_array.min():.4f}/{output_array.max():.4f}")
        
        print("✅ Server test successful!")
        return True
        
    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        channel.close()

def test_server_health(host="localhost", port=8180):
    """Test basic server connectivity."""
    
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        grpc.channel_ready_future(channel).result(timeout=5)
        print(f"✅ Server at {host}:{port} is reachable")
        channel.close()
        return True
    except Exception as e:
        print(f"❌ Cannot connect to server at {host}:{port}: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Jetson Single Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8180, help="Server port")
    parser.add_argument("--model", default="EENetB0_1_FC", help="Model name")
    parser.add_argument("--health-only", action="store_true", help="Only test connectivity")
    
    args = parser.parse_args()
    
    print("=== Jetson Single Server Test ===")
    
    # Test basic connectivity first
    if not test_server_health(args.host, args.port):
        print("Server is not reachable. Make sure it's running.")
        exit(1)
    
    if not args.health_only:
        # Test with actual request
        test_server(args.host, args.port, args.model)
    
    print("=== Test Complete ===") 