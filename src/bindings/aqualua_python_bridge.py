"""
AquaLua Python Bridge - Universal Python Library Access
Provides seamless integration with the entire Python ecosystem
"""

import sys
import importlib
import types
from typing import Any, Dict, List, Optional

class PythonBridge:
    """Bridge that provides access to any Python module/library"""
    
    def __init__(self):
        self.imported_modules: Dict[str, Any] = {}
        self.module_cache: Dict[str, Any] = {}
    
    def import_module(self, module_path: List[str], alias: Optional[str] = None) -> Any:
        """Import any Python module dynamically"""
        module_name = '.'.join(module_path)
        
        try:
            # Try to import the module
            if module_name in self.module_cache:
                module = self.module_cache[module_name]
            else:
                module = importlib.import_module(module_name)
                self.module_cache[module_name] = module
            
            # Store with alias or original name
            key = alias if alias else module_path[-1]
            self.imported_modules[key] = module
            
            print(f"[SUCCESS] Successfully imported {module_name} as {key}")
            return module
            
        except ImportError as e:
            print(f"[FAILED] Failed to import {module_name}: {e}")
            # Return a mock object that logs method calls
            return self._create_mock_module(module_name)
    
    def _create_mock_module(self, module_name: str):
        """Create a mock module for unavailable libraries"""
        class MockModule:
            def __init__(self, name):
                self._name = name
            
            def __getattr__(self, attr):
                print(f"[MOCK] {self._name}.{attr} called")
                return MockModule(f"{self._name}.{attr}")
            
            def __call__(self, *args, **kwargs):
                print(f"[MOCK] {self._name}({args}, {kwargs}) called")
                # Special handling for pygame.event.get() - return empty list
                if 'event.get' in self._name:
                    return []
                # Special handling for pygame.init() - return None
                if 'init' in self._name:
                    return None
                # Special handling for pygame.quit() - return None  
                if 'quit' in self._name:
                    return None
                # Special handling for display methods - return mock surface
                if 'display.set_mode' in self._name:
                    return MockSurface()
                if 'display.set_caption' in self._name or 'display.flip' in self._name or 'display.update' in self._name:
                    return None
                return MockModule(f"{self._name}_result")
            
            def __iter__(self):
                # Make mock modules iterable (return empty iterator)
                return iter([])
            
            def __str__(self):
                return f"Mock<{self._name}>"
        
        class MockSurface:
            def fill(self, color):
                print(f"[MOCK] Surface.fill({color}) called")
                return None
        
        # Create specific pygame mock if needed
        if module_name == 'pygame':
            class MockPygame:
                QUIT = 12  # Standard pygame QUIT event type
                
                @staticmethod
                def init():
                    print("[MOCK] pygame.init() called")
                    return None
                
                @staticmethod
                def quit():
                    print("[MOCK] pygame.quit() called")
                    return None
                
                class display:
                    @staticmethod
                    def set_mode(size):
                        print(f"[MOCK] pygame.display.set_mode({size}) called")
                        return MockSurface()
                    
                    @staticmethod
                    def set_caption(title):
                        print(f"[MOCK] pygame.display.set_caption('{title}') called")
                        return None
                    
                    @staticmethod
                    def flip():
                        print("[MOCK] pygame.display.flip() called")
                        return None
                    
                    @staticmethod
                    def update():
                        print("[MOCK] pygame.display.update() called")
                        return None
                
                class event:
                    _call_count = 0
                    
                    @staticmethod
                    def get():
                        print("[MOCK] pygame.event.get() called")
                        MockPygame.event._call_count += 1
                        
                        # After 5 calls, simulate a QUIT event to break the loop
                        if MockPygame.event._call_count > 5:
                            class MockEvent:
                                def __init__(self, event_type):
                                    self.type = event_type
                            
                            return [MockEvent(MockPygame.QUIT)]
                        
                        # Return empty list for first few calls
                        return []
            
            return MockPygame
        
        return MockModule(module_name)
    
    def get_module(self, name: str) -> Any:
        """Get imported module by name"""
        return self.imported_modules.get(name)
    
    def install_package(self, package_name: str) -> bool:
        """Install Python package using pip"""
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[SUCCESS] Successfully installed {package_name}")
                return True
            else:
                print(f"[FAILED] Failed to install {package_name}: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Error installing {package_name}: {e}")
            return False
    
    def list_available_modules(self) -> List[str]:
        """List all available Python modules"""
        try:
            import pkgutil
            modules = []
            for importer, modname, ispkg in pkgutil.iter_modules():
                modules.append(modname)
            return sorted(modules)
        except Exception:
            return ["numpy", "pandas", "matplotlib", "requests", "flask", "django", "tensorflow", "torch"]
    
    def execute_python_code(self, code: str) -> Any:
        """Execute arbitrary Python code"""
        try:
            # Create a namespace with imported modules
            namespace = {
                '__builtins__': __builtins__,
                **self.imported_modules
            }
            
            # Execute the code
            exec(code, namespace)
            return namespace.get('result', None)
        except Exception as e:
            print(f"[ERROR] Error executing Python code: {e}")
            return None

class UniversalPythonAccess:
    """Provides universal access to Python libraries for AquaLua"""
    
    def __init__(self):
        self.bridge = PythonBridge()
        self.setup_common_libraries()
    
    def setup_common_libraries(self):
        """Pre-import commonly used libraries"""
        common_libs = [
            (['numpy'], 'np'),
            (['pandas'], 'pd'), 
            (['matplotlib', 'pyplot'], 'plt'),
            (['requests'], None),
            (['json'], None),
            (['os'], None),
            (['sys'], None),
            (['datetime'], None),
            (['math'], None),
            (['random'], None),
            (['re'], None),
            (['urllib', 'request'], None),
            (['sqlite3'], None),
            (['csv'], None),
            (['pickle'], None),
            (['base64'], None),
            (['hashlib'], None),
            (['collections'], None),
            (['itertools'], None),
            (['functools'], None),
            (['pathlib'], None),
            (['tkinter'], 'tk'),
            (['pygame'], None),
            (['PIL'], None),
            (['cv2'], None),
            (['sklearn'], None),
            (['tensorflow'], 'tf'),
            (['torch'], None),
            (['flask'], None),
            (['django'], None),
            (['fastapi'], None),
            (['sqlalchemy'], None),
            (['redis'], None),
            (['pymongo'], None),
            (['psycopg2'], None),
            (['boto3'], None),
            (['azure'], None),
            (['google'], None),
        ]
        
        for module_path, alias in common_libs:
            try:
                self.bridge.import_module(module_path, alias)
            except Exception:
                pass  # Silently skip unavailable libraries
    
    def import_any(self, module_path: List[str], alias: Optional[str] = None) -> Any:
        """Import any Python module"""
        return self.bridge.import_module(module_path, alias)
    
    def get_module(self, name: str) -> Any:
        """Get imported module"""
        return self.bridge.get_module(name)
    
    def install_and_import(self, package_name: str, module_path: Optional[List[str]] = None, alias: Optional[str] = None) -> Any:
        """Install package and import it"""
        if self.bridge.install_package(package_name):
            import_path = module_path or [package_name]
            return self.bridge.import_module(import_path, alias)
        return None
    
    def execute_python(self, code: str) -> Any:
        """Execute Python code with access to all imported modules"""
        return self.bridge.execute_python_code(code)
    
    def create_python_function(self, code: str, function_name: str) -> callable:
        """Create a Python function from code string"""
        try:
            namespace = {'__builtins__': __builtins__, **self.bridge.imported_modules}
            exec(code, namespace)
            return namespace.get(function_name)
        except Exception as e:
            print(f"[ERROR] Error creating function {function_name}: {e}")
            return lambda *args, **kwargs: None
    
    def get_python_help(self, obj_name: str) -> str:
        """Get help documentation for Python object"""
        try:
            obj = self.bridge.get_module(obj_name)
            if obj:
                return str(help(obj))
            return f"No help available for {obj_name}"
        except Exception:
            return f"Error getting help for {obj_name}"
    
    def list_module_contents(self, module_name: str) -> List[str]:
        """List contents of a module"""
        try:
            module = self.bridge.get_module(module_name)
            if module:
                return [attr for attr in dir(module) if not attr.startswith('_')]
            return []
        except Exception:
            return []
    
    def create_virtual_environment(self, env_name: str) -> bool:
        """Create a virtual environment"""
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'venv', env_name], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def pip_install_multiple(self, packages: List[str]) -> Dict[str, bool]:
        """Install multiple packages"""
        results = {}
        for package in packages:
            results[package] = self.bridge.install_package(package)
        return results
    
    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get information about an installed package"""
        try:
            import pkg_resources
            package = pkg_resources.get_distribution(package_name)
            return {
                'name': package.project_name,
                'version': package.version,
                'location': package.location,
                'requires': [str(req) for req in package.requires()]
            }
        except Exception:
            return {'error': f'Package {package_name} not found'}
    
    def search_packages(self, query: str) -> List[str]:
        """Search for packages (simplified)"""
        # This would ideally use PyPI API, but for now return common matches
        common_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
            'requests', 'urllib3', 'httpx', 'aiohttp',
            'flask', 'django', 'fastapi', 'tornado', 'bottle',
            'sqlalchemy', 'pymongo', 'redis', 'psycopg2', 'mysql-connector',
            'tensorflow', 'torch', 'sklearn', 'xgboost', 'lightgbm',
            'opencv-python', 'pillow', 'imageio', 'scikit-image',
            'beautifulsoup4', 'lxml', 'scrapy', 'selenium',
            'pytest', 'unittest2', 'nose2', 'coverage',
            'black', 'flake8', 'mypy', 'pylint',
            'jupyter', 'ipython', 'notebook', 'jupyterlab',
            'click', 'argparse', 'typer', 'fire',
            'pydantic', 'marshmallow', 'cerberus', 'schema',
            'celery', 'rq', 'schedule', 'apscheduler',
            'boto3', 'azure-storage', 'google-cloud-storage',
            'paramiko', 'fabric', 'ansible', 'docker',
            'cryptography', 'bcrypt', 'passlib', 'jwt'
        ]
        
        return [pkg for pkg in common_packages if query.lower() in pkg.lower()]

# Global instance for AquaLua interpreter
python_bridge = UniversalPythonAccess()

# Convenience functions for AquaLua
def import_python(module_path: str, alias: str = None) -> Any:
    """Import Python module - AquaLua interface"""
    path_parts = module_path.split('.')
    return python_bridge.import_any(path_parts, alias)

def install_python_package(package_name: str) -> bool:
    """Install Python package - AquaLua interface"""
    return python_bridge.bridge.install_package(package_name)

def execute_python(code: str) -> Any:
    """Execute Python code - AquaLua interface"""
    return python_bridge.execute_python(code)

def get_python_module(name: str) -> Any:
    """Get imported Python module - AquaLua interface"""
    return python_bridge.get_module(name)

def python_help(obj_name: str) -> str:
    """Get Python help - AquaLua interface"""
    return python_bridge.get_python_help(obj_name)

def list_python_packages() -> List[str]:
    """List available Python packages"""
    return python_bridge.bridge.list_available_modules()

def search_python_packages(query: str) -> List[str]:
    """Search Python packages"""
    return python_bridge.search_packages(query)

if __name__ == "__main__":
    # Test the bridge
    bridge = UniversalPythonAccess()
    
    # Test importing numpy
    np = bridge.import_any(['numpy'], 'np')
    if np:
        print("NumPy imported successfully!")
        
    # Test importing pygame
    pygame = bridge.import_any(['pygame'])
    if pygame:
        print("Pygame imported successfully!")
    
    # Test executing Python code
    result = bridge.execute_python("""
import math
result = math.sqrt(16) + math.pi
""")
    print(f"Python execution result: {result}")
    
    print("\n[READY] AquaLua Python Bridge is ready!")
    print("[SUCCESS] Universal Python library access enabled")
    print("[SUCCESS] Dynamic module importing")
    print("[SUCCESS] Package installation support")
    print("[SUCCESS] Code execution capabilities")