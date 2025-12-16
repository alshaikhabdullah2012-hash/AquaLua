"""
Aqualua Backend - Python FFI bridge to C runtime
Provides high-level Python interface to the optimized C backend
"""

import ctypes
import numpy as np
from numpy import ctypeslib as np_ctypes
from ctypes import POINTER, c_int, c_float, c_size_t, c_void_p, Structure
from typing import List, Tuple, Optional, Union
import os
import platform

# Load the C runtime library
def load_runtime_library():
    """Load the compiled Aqualua C runtime library"""
    system = platform.system().lower()
    
    if system == "windows":
        lib_name = "aqualua_runtime.dll"
    elif system == "darwin":  # macOS
        lib_name = "libaqualua_runtime.dylib"
    else:  # Linux and others
        lib_name = "libaqualua_runtime.so"
    
    # Try multiple locations
    search_paths = [
        os.path.join(os.path.dirname(__file__), lib_name),  # Same directory as this file
        os.path.join(os.getcwd(), lib_name),  # Current working directory
        lib_name  # System PATH
    ]
    
    for lib_path in search_paths:
        try:
            print(f"Trying to load C runtime from: {lib_path}")
            lib = ctypes.CDLL(lib_path)
            print(f"[OK] Successfully loaded C runtime: {lib_path}")
            return lib
        except OSError as e:
            print(f"[X] Failed to load from {lib_path}: {e}")
            continue
    
    print("[WARN] C runtime not found, falling back to Python implementation")
    return None

# Attempt to load C runtime
try:
    runtime_lib = load_runtime_library()
    HAS_C_BACKEND = runtime_lib is not None
    if HAS_C_BACKEND:
        print("[SUCCESS] Aqualua C backend loaded successfully!")
    else:
        print("[FALLBACK] Using Python fallback implementation")
except Exception as e:
    print(f"Failed to load C backend: {e}")
    HAS_C_BACKEND = False
    runtime_lib = None

# Data type mappings
class DataType:
    F32 = 0
    F64 = 1
    F16 = 2
    I32 = 3
    I64 = 4
    BOOL = 5

class DeviceType:
    CPU = 0
    GPU = 1
    AUTO = 2

# C structure definitions for FFI
class CTensor(Structure):
    """C tensor structure mirror"""
    _fields_ = [
        ("data", c_void_p),
        ("shape", POINTER(c_int)),
        ("ndim", c_int),
        ("dtype", c_int),
        ("device", c_int),
        ("size", c_size_t),
        ("bytes", c_size_t),
        ("requires_grad", ctypes.c_bool),
        ("grad", c_void_p),
        ("ref_count", c_int),
    ]

class CLinearLayer(Structure):
    """C linear layer structure mirror"""
    _fields_ = [
        ("weight", c_void_p),
        ("bias", c_void_p),
        ("in_features", c_int),
        ("out_features", c_int),
    ]

# Configure C function signatures if backend is available
if HAS_C_BACKEND:
    # Tensor creation functions
    runtime_lib.py_tensor_zeros.argtypes = [POINTER(c_int), c_int, c_int, c_int]
    runtime_lib.py_tensor_zeros.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_ones.argtypes = [POINTER(c_int), c_int, c_int, c_int]
    runtime_lib.py_tensor_ones.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_random.argtypes = [POINTER(c_int), c_int, c_int, c_int]
    runtime_lib.py_tensor_random.restype = POINTER(CTensor)
    
    # Tensor operations
    runtime_lib.py_tensor_add.argtypes = [POINTER(CTensor), POINTER(CTensor)]
    runtime_lib.py_tensor_add.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_multiply.argtypes = [POINTER(CTensor), POINTER(CTensor)]
    runtime_lib.py_tensor_multiply.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_matmul.argtypes = [POINTER(CTensor), POINTER(CTensor)]
    runtime_lib.py_tensor_matmul.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_relu.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_relu.restype = POINTER(CTensor)
    
    runtime_lib.py_tensor_sigmoid.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_sigmoid.restype = POINTER(CTensor)
    
    # Neural network layers
    runtime_lib.py_linear_create.argtypes = [c_int, c_int, c_int]
    runtime_lib.py_linear_create.restype = POINTER(CLinearLayer)
    
    runtime_lib.py_linear_forward.argtypes = [POINTER(CLinearLayer), POINTER(CTensor)]
    runtime_lib.py_linear_forward.restype = POINTER(CTensor)
    
    # Memory management
    runtime_lib.py_tensor_destroy.argtypes = [POINTER(CTensor)]
    runtime_lib.py_linear_destroy.argtypes = [POINTER(CLinearLayer)]
    
    # Data access
    runtime_lib.py_tensor_data_f32.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_data_f32.restype = POINTER(c_float)
    
    runtime_lib.py_tensor_shape.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_shape.restype = POINTER(c_int)
    
    runtime_lib.py_tensor_ndim.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_ndim.restype = c_int
    
    runtime_lib.py_tensor_size.argtypes = [POINTER(CTensor)]
    runtime_lib.py_tensor_size.restype = c_size_t
    
    # Device management
    runtime_lib.py_device_init.argtypes = []
    runtime_lib.py_device_auto_select.restype = c_int

class AqualuaTensor:
    """High-level Python tensor interface"""
    
    def __init__(self, shape: List[int], dtype: int = DataType.F32, device: int = DeviceType.AUTO, data=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._c_tensor = None
        self._numpy_data = None
        
        if HAS_C_BACKEND:
            # Use C backend for performance
            shape_array = (c_int * len(shape))(*shape)
            if data is None:
                self._c_tensor = runtime_lib.py_tensor_zeros(shape_array, len(shape), dtype, device)
            else:
                # TODO: Create tensor with data
                self._c_tensor = runtime_lib.py_tensor_zeros(shape_array, len(shape), dtype, device)
                self._copy_data_to_c(data)
        else:
            # Fallback to NumPy for development
            if data is None:
                self._numpy_data = np.zeros(shape, dtype=np.float32)
            else:
                self._numpy_data = np.array(data, dtype=np.float32).reshape(shape)
    
    def _copy_data_to_c(self, data):
        """Copy Python data to C tensor"""
        if not HAS_C_BACKEND or not self._c_tensor:
            return
        
        # Get C tensor data pointer
        c_data_ptr = runtime_lib.py_tensor_data_f32(self._c_tensor)
        size = runtime_lib.py_tensor_size(self._c_tensor)
        
        # Convert to numpy array and copy
        np_data = np.array(data, dtype=np.float32).flatten()
        ctypes.memmove(c_data_ptr, np_data.ctypes.data, size * 4)  # 4 bytes per float32
    
    def to_numpy(self) -> np.ndarray:
        """Convert tensor to NumPy array"""
        if HAS_C_BACKEND and self._c_tensor:
            # Copy data from C tensor
            c_data_ptr = runtime_lib.py_tensor_data_f32(self._c_tensor)
            size = runtime_lib.py_tensor_size(self._c_tensor)
            shape_ptr = runtime_lib.py_tensor_shape(self._c_tensor)
            ndim = runtime_lib.py_tensor_ndim(self._c_tensor)
            
            # Extract shape
            shape = [shape_ptr[i] for i in range(ndim)]
            
            # Create numpy array from C data
            np_array = np_ctypes.as_array(c_data_ptr, shape=(size,))
            return np_array.reshape(shape).copy()
        else:
            return self._numpy_data.copy()
    
    def __add__(self, other):
        """Tensor addition"""
        if HAS_C_BACKEND and self._c_tensor and other._c_tensor:
            result_c = runtime_lib.py_tensor_add(self._c_tensor, other._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            result.shape = self.shape
            result.dtype = self.dtype
            result.device = self.device
            return result
        else:
            # NumPy fallback
            result_data = self._numpy_data + other._numpy_data
            return AqualuaTensor(self.shape, self.dtype, self.device, result_data)
    
    def __mul__(self, other):
        """Element-wise multiplication"""
        if HAS_C_BACKEND and self._c_tensor and other._c_tensor:
            result_c = runtime_lib.py_tensor_multiply(self._c_tensor, other._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            result.shape = self.shape
            result.dtype = self.dtype
            result.device = self.device
            return result
        else:
            result_data = self._numpy_data * other._numpy_data
            return AqualuaTensor(self.shape, self.dtype, self.device, result_data)
    
    def __matmul__(self, other):
        """Matrix multiplication"""
        if HAS_C_BACKEND and self._c_tensor and other._c_tensor:
            result_c = runtime_lib.py_tensor_matmul(self._c_tensor, other._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            # Calculate result shape for matrix multiplication
            result_shape = self.shape[:-1] + other.shape[-1:]
            result.shape = result_shape
            result.dtype = self.dtype
            result.device = self.device
            return result
        else:
            result_data = self._numpy_data @ other._numpy_data
            return AqualuaTensor(result_data.shape, self.dtype, self.device, result_data)
    
    def relu(self):
        """ReLU activation function"""
        if HAS_C_BACKEND and self._c_tensor:
            result_c = runtime_lib.py_tensor_relu(self._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            result.shape = self.shape
            result.dtype = self.dtype
            result.device = self.device
            return result
        else:
            result_data = np.maximum(0, self._numpy_data)
            return AqualuaTensor(self.shape, self.dtype, self.device, result_data)
    
    def sigmoid(self):
        """Sigmoid activation function"""
        if HAS_C_BACKEND and self._c_tensor:
            result_c = runtime_lib.py_tensor_sigmoid(self._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            result.shape = self.shape
            result.dtype = self.dtype
            result.device = self.device
            return result
        else:
            result_data = 1.0 / (1.0 + np.exp(-self._numpy_data))
            return AqualuaTensor(self.shape, self.dtype, self.device, result_data)
    
    def __del__(self):
        """Cleanup C resources"""
        if HAS_C_BACKEND and self._c_tensor:
            runtime_lib.py_tensor_destroy(self._c_tensor)
    
    def __repr__(self):
        return f"AqualuaTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

class Linear:
    """Linear/Dense layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, device: int = DeviceType.AUTO):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self._c_layer = None
        
        if HAS_C_BACKEND:
            self._c_layer = runtime_lib.py_linear_create(in_features, out_features, device)
        else:
            # NumPy fallback - Xavier initialization
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.weight = AqualuaTensor([out_features, in_features], device=device, 
                                     data=np.random.randn(out_features, in_features) * scale)
            self.bias = AqualuaTensor([out_features], device=device, 
                                    data=np.zeros(out_features))
    
    def __call__(self, x: AqualuaTensor) -> AqualuaTensor:
        """Forward pass"""
        if HAS_C_BACKEND and self._c_layer:
            result_c = runtime_lib.py_linear_forward(self._c_layer, x._c_tensor)
            result = AqualuaTensor.__new__(AqualuaTensor)
            result._c_tensor = result_c
            result.shape = [x.shape[0], self.out_features]  # Assuming batch dimension
            result.dtype = x.dtype
            result.device = x.device
            return result
        else:
            # NumPy fallback: output = input @ weight.T + bias
            output = x @ self.weight
            return output + self.bias
    
    def __del__(self):
        """Cleanup C resources"""
        if HAS_C_BACKEND and self._c_layer:
            runtime_lib.py_linear_destroy(self._c_layer)

# Tensor creation functions
def zeros(shape: List[int], dtype: int = DataType.F32, device: int = DeviceType.AUTO) -> AqualuaTensor:
    """Create tensor filled with zeros"""
    return AqualuaTensor(shape, dtype, device)

def ones(shape: List[int], dtype: int = DataType.F32, device: int = DeviceType.AUTO) -> AqualuaTensor:
    """Create tensor filled with ones"""
    if HAS_C_BACKEND:
        shape_array = (c_int * len(shape))(*shape)
        c_tensor = runtime_lib.py_tensor_ones(shape_array, len(shape), dtype, device)
        result = AqualuaTensor.__new__(AqualuaTensor)
        result._c_tensor = c_tensor
        result.shape = shape
        result.dtype = dtype
        result.device = device
        return result
    else:
        return AqualuaTensor(shape, dtype, device, np.ones(shape))

def random(shape: List[int], dtype: int = DataType.F32, device: int = DeviceType.AUTO) -> AqualuaTensor:
    """Create tensor with random values"""
    if HAS_C_BACKEND:
        shape_array = (c_int * len(shape))(*shape)
        c_tensor = runtime_lib.py_tensor_random(shape_array, len(shape), dtype, device)
        result = AqualuaTensor.__new__(AqualuaTensor)
        result._c_tensor = c_tensor
        result.shape = shape
        result.dtype = dtype
        result.device = device
        return result
    else:
        return AqualuaTensor(shape, dtype, device, np.random.randn(*shape))

def tensor(data, dtype: int = DataType.F32, device: int = DeviceType.AUTO) -> AqualuaTensor:
    """Create tensor from data"""
    np_data = np.array(data)
    return AqualuaTensor(list(np_data.shape), dtype, device, np_data)

# Activation functions
def relu(x: AqualuaTensor) -> AqualuaTensor:
    """ReLU activation function"""
    return x.relu()

def sigmoid(x: AqualuaTensor) -> AqualuaTensor:
    """Sigmoid activation function"""
    return x.sigmoid()

# Loss functions
def cross_entropy(predictions: AqualuaTensor, targets: AqualuaTensor) -> AqualuaTensor:
    """Cross entropy loss"""
    # Simplified implementation - in production would use C backend
    pred_np = predictions.to_numpy()
    target_np = targets.to_numpy()
    
    # Add epsilon for numerical stability
    pred_np = np.clip(pred_np, 1e-8, 1.0 - 1e-8)
    loss = -np.sum(target_np * np.log(pred_np)) / pred_np.shape[0]
    
    return tensor([loss])

def mse(predictions: AqualuaTensor, targets: AqualuaTensor) -> AqualuaTensor:
    """Mean squared error loss"""
    diff = predictions + (targets * tensor([-1.0]))  # predictions - targets
    squared = diff * diff
    return tensor([squared.to_numpy().mean()])

# Device management
def device_init():
    """Initialize device subsystem"""
    if HAS_C_BACKEND:
        runtime_lib.py_device_init()
    print("Aqualua backend initialized")

def device_auto_select() -> int:
    """Auto-select best available device"""
    if HAS_C_BACKEND:
        return runtime_lib.py_device_auto_select()
    return DeviceType.CPU

# Initialize backend on import
if HAS_C_BACKEND:
    device_init()

# Export public API
__all__ = [
    'AqualuaTensor', 'Linear', 'DataType', 'DeviceType',
    'zeros', 'ones', 'random', 'tensor',
    'relu', 'sigmoid', 'cross_entropy', 'mse',
    'device_init', 'device_auto_select',
    'HAS_C_BACKEND'
]