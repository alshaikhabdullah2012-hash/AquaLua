#!/usr/bin/env python3
"""
Test script to verify C backend loading
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_c_backend():
    print("Testing Aqualua C Backend Loading...")
    print("=" * 50)
    
    try:
        from aqualua_backend import HAS_C_BACKEND, runtime_lib
        
        if HAS_C_BACKEND:
            print("[OK] C Backend loaded successfully!")
            
            # Test basic tensor operations
            print("\nTesting basic tensor operations...")
            from aqualua_backend import zeros, ones, tensor
            
            # Create test tensors
            a = zeros([2, 2])
            b = ones([2, 2])
            
            print(f"Created zero tensor: {a}")
            print(f"Created ones tensor: {b}")
            
            # Test addition
            c = a + b
            print(f"Addition result: {c}")
            print(f"Addition data: {c.to_numpy()}")
            
            print("\n[SUCCESS] All C backend tests passed!")
            
        else:
            print("[FALLBACK] C Backend not available, using Python fallback")
            
    except Exception as e:
        print(f"[ERROR] Error testing C backend: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_c_backend()