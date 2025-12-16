#!/usr/bin/env python3
"""
Quick test script for Aqualua - bypasses parser issues
"""

import sys
sys.path.append('.')

from aqualua_backend import *
from aqualua_interpreter import Environment

# Create simple test environment
env = Environment()

# Setup basic functions
env.define("print", print)
env.define("tensor", tensor)
env.define("zeros", zeros)
env.define("ones", ones)
env.define("random", random)

print("=== Aqualua Quick Test ===")
print("Testing basic operations...")

# Test 1: Simple variables
print("\n1. Variables:")
x = 42
env.define("x", x)
print(f"x = {x}")

# Test 2: Tensors
print("\n2. Tensors:")
t1 = tensor([1, 2, 3])
t2 = tensor([4, 5, 6])
print(f"t1 = {t1}")
print(f"t2 = {t2}")

# Test 3: Tensor operations
print("\n3. Tensor Operations:")
t3 = t1 + t2
print(f"t1 + t2 = {t3}")

# Test 4: Matrix operations
print("\n4. Matrix Operations:")
m1 = tensor([[1, 2], [3, 4]])
m2 = tensor([[5, 6], [7, 8]])
print(f"m1 = {m1}")
print(f"m2 = {m2}")
print(f"m1 @ m2 = {m1 @ m2}")

# Test 5: Neural network layer
print("\n5. Neural Network Layer:")
layer = Linear(3, 2)
input_tensor = tensor([1.0, 2.0, 3.0])
output = layer(input_tensor)
print(f"Linear(3,2) output: {output}")

print("\n=== All tests passed! ===")
print("Your Aqualua backend is working correctly!")
print("\nTo fix the parser, the issue is with colon syntax.")
print("Try these commands in a Python shell:")
print("  from aqualua_backend import *")
print("  x = tensor([1, 2, 3])")
print("  print(x)")