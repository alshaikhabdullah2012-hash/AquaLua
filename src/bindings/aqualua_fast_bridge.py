"""
AquaLua Fast Bridge - Cached bytecode for zero-overhead Python calls
"""

import importlib
import weakref
from typing import Dict, Any, Callable, Optional

class FastPythonBridge:
    """Ultra-fast Python bridge with function caching"""
    
    def __init__(self):
        self.function_cache: Dict[str, Callable] = {}
        self.module_cache: Dict[str, Any] = {}
        self.attribute_cache: Dict[str, Any] = {}
        
    def get_cached_function(self, module_path: str, func_name: str) -> Callable:
        """Get cached Python function for direct calls"""
        cache_key = f"{module_path}.{func_name}"
        
        if cache_key in self.function_cache:
            return self.function_cache[cache_key]
        
        # Load and cache function
        try:
            module = self._get_module(module_path)
            func = getattr(module, func_name)
            self.function_cache[cache_key] = func
            return func
        except Exception:
            # Return no-op function for missing functions
            def noop(*args, **kwargs):
                return None
            self.function_cache[cache_key] = noop
            return noop
    
    def get_cached_attribute(self, module_path: str, attr_name: str) -> Any:
        """Get cached Python attribute/constant"""
        cache_key = f"{module_path}.{attr_name}"
        
        if cache_key in self.attribute_cache:
            return self.attribute_cache[cache_key]
        
        try:
            module = self._get_module(module_path)
            attr = getattr(module, attr_name)
            self.attribute_cache[cache_key] = attr
            return attr
        except Exception:
            self.attribute_cache[cache_key] = None
            return None
    
    def _get_module(self, module_path: str) -> Any:
        """Get cached Python module"""
        if module_path in self.module_cache:
            return self.module_cache[module_path]
        
        try:
            module = importlib.import_module(module_path)
            self.module_cache[module_path] = module
            return module
        except ImportError:
            # Create mock module
            class MockModule:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            mock = MockModule()
            self.module_cache[module_path] = mock
            return mock
    
    def call_fast(self, module_path: str, func_name: str, args: list, kwargs: dict = None) -> Any:
        """Ultra-fast function call with caching"""
        func = self.get_cached_function(module_path, func_name)
        return func(*args, **(kwargs or {}))

# Global bridge instance
fast_bridge = FastPythonBridge()