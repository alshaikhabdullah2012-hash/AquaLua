"""
AquaLua Library Transpiler - Converts Python libraries to AquaLua bindings
"""

import ast
import inspect
import importlib
import types
from typing import Dict, List, Any, Optional
import os

class LibraryTranspiler:
    def __init__(self):
        self.bindings = {}
        self.type_mappings = {
            'int': 'i32',
            'float': 'f32', 
            'str': 'string',
            'bool': 'bool',
            'list': 'List',
            'dict': 'Dict',
            'tuple': 'Tuple'
        }
    
    def transpile_library(self, library_name: str) -> str:
        """Transpile a Python library to AquaLua bindings"""
        print(f"[TRANSPILE] Analyzing {library_name}...")
        
        try:
            # Import the library
            lib = importlib.import_module(library_name)
            
            # Generate AquaLua bindings
            bindings = self._analyze_module(lib, library_name)
            
            # Generate AquaLua code
            aq_code = self._generate_aqualua_code(library_name, bindings)
            
            # Save bindings file
            binding_file = f"{library_name}_bindings.aq"
            with open(binding_file, 'w') as f:
                f.write(aq_code)
            
            print(f"[SUCCESS] Generated {binding_file}")
            return aq_code
            
        except Exception as e:
            print(f"[ERROR] Failed to transpile {library_name}: {e}")
            return self._generate_mock_bindings(library_name)
    
    def _analyze_module(self, module, name: str) -> Dict:
        """Analyze Python module and extract API"""
        bindings = {
            'constants': {},
            'functions': {},
            'classes': {},
            'submodules': {}
        }
        
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            try:
                attr = getattr(module, attr_name)
                
                if inspect.ismodule(attr):
                    # Submodule
                    bindings['submodules'][attr_name] = self._analyze_module(attr, f"{name}.{attr_name}")
                
                elif inspect.isclass(attr):
                    # Class
                    bindings['classes'][attr_name] = self._analyze_class(attr)
                
                elif inspect.isfunction(attr) or inspect.isbuiltin(attr):
                    # Function
                    bindings['functions'][attr_name] = self._analyze_function(attr)
                
                elif not callable(attr):
                    # Constant
                    bindings['constants'][attr_name] = self._analyze_constant(attr)
                    
            except Exception:
                # Skip problematic attributes
                continue
        
        return bindings
    
    def _analyze_class(self, cls) -> Dict:
        """Analyze Python class"""
        class_info = {
            'methods': {},
            'static_methods': {},
            'properties': {},
            'constructor': None
        }
        
        # Analyze constructor
        if hasattr(cls, '__init__'):
            class_info['constructor'] = self._analyze_function(cls.__init__)
        
        # Analyze methods
        for method_name in dir(cls):
            if method_name.startswith('_') and method_name != '__init__':
                continue
                
            try:
                method = getattr(cls, method_name)
                if inspect.ismethod(method) or inspect.isfunction(method):
                    class_info['methods'][method_name] = self._analyze_function(method)
                elif isinstance(method, staticmethod):
                    class_info['static_methods'][method_name] = self._analyze_function(method)
                elif isinstance(method, property):
                    class_info['properties'][method_name] = {'type': 'any'}
            except Exception:
                continue
        
        return class_info
    
    def _analyze_function(self, func) -> Dict:
        """Analyze Python function"""
        func_info = {
            'params': [],
            'return_type': 'any',
            'is_builtin': inspect.isbuiltin(func)
        }
        
        try:
            if not inspect.isbuiltin(func):
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    param_info = {
                        'name': param_name,
                        'type': self._map_type(param.annotation) if param.annotation != param.empty else 'any',
                        'default': param.default if param.default != param.empty else None
                    }
                    func_info['params'].append(param_info)
                
                if sig.return_annotation != sig.empty:
                    func_info['return_type'] = self._map_type(sig.return_annotation)
        except Exception:
            pass
        
        return func_info
    
    def _analyze_constant(self, value) -> Dict:
        """Analyze Python constant"""
        return {
            'type': self._map_type(type(value).__name__),
            'value': value
        }
    
    def _map_type(self, python_type) -> str:
        """Map Python type to AquaLua type"""
        if isinstance(python_type, str):
            return self.type_mappings.get(python_type, python_type)
        elif hasattr(python_type, '__name__'):
            return self.type_mappings.get(python_type.__name__, python_type.__name__)
        else:
            return 'any'
    
    def _generate_aqualua_code(self, lib_name: str, bindings: Dict) -> str:
        """Generate AquaLua binding code"""
        code = f"""# AquaLua bindings for {lib_name}
# Auto-generated by AquaLua Transpiler

import python_bridge

"""
        
        # Generate constants
        if bindings['constants']:
            code += f"# {lib_name} Constants\n"
            for name, const_info in bindings['constants'].items():
                code += f"let {lib_name}.{name} = python_bridge.get_constant(\"{lib_name}\", \"{name}\")\n"
            code += "\n"
        
        # Generate functions
        if bindings['functions']:
            code += f"# {lib_name} Functions\n"
            for name, func_info in bindings['functions'].items():
                params = ", ".join([p['name'] for p in func_info['params']])
                code += f"fn {lib_name}.{name}({params}):\n"
                code += f"    return python_bridge.call_function(\"{lib_name}\", \"{name}\", [{params}])\n\n"
        
        # Generate classes
        if bindings['classes']:
            code += f"# {lib_name} Classes\n"
            for class_name, class_info in bindings['classes'].items():
                code += f"class {lib_name}.{class_name}:\n"
                
                # Constructor
                if class_info['constructor']:
                    params = ", ".join([p['name'] for p in class_info['constructor']['params']])
                    code += f"    fn __init__({params}):\n"
                    code += f"        self._py_obj = python_bridge.create_object(\"{lib_name}\", \"{class_name}\", [{params}])\n\n"
                
                # Methods
                for method_name, method_info in class_info['methods'].items():
                    if method_name == '__init__':
                        continue
                    params = ", ".join([p['name'] for p in method_info['params']])
                    code += f"    fn {method_name}({params}):\n"
                    code += f"        return python_bridge.call_method(self._py_obj, \"{method_name}\", [{params}])\n\n"
                
                code += "\n"
        
        # Generate submodules
        if bindings['submodules']:
            code += f"# {lib_name} Submodules\n"
            for submod_name, submod_bindings in bindings['submodules'].items():
                submod_code = self._generate_aqualua_code(f"{lib_name}.{submod_name}", submod_bindings)
                code += submod_code
        
        return code
    
    def _generate_mock_bindings(self, lib_name: str) -> str:
        """Generate mock bindings for unavailable libraries"""
        return f"""# Mock bindings for {lib_name}
# Library not available, using mock implementation

import python_bridge

# Mock module that logs calls
let {lib_name} = python_bridge.create_mock_module(\"{lib_name}\")
"""

class AquaLuaPackageManager:
    """Package manager for AquaLua libraries"""
    
    def __init__(self):
        self.transpiler = LibraryTranspiler()
        self.installed_packages = set()
        self.bindings_dir = "aqualua_bindings"
        
        # Create bindings directory
        os.makedirs(self.bindings_dir, exist_ok=True)
    
    def install_package(self, package_name: str) -> bool:
        """Install Python package and generate AquaLua bindings"""
        print(f"[INSTALL] Installing {package_name}...")
        
        # First, try to install via pip
        if not self._pip_install(package_name):
            print(f"[WARNING] Could not install {package_name} via pip, creating mock bindings")
        
        # Generate AquaLua bindings
        bindings_code = self.transpiler.transpile_library(package_name)
        
        # Save bindings
        bindings_file = os.path.join(self.bindings_dir, f"{package_name}.aq")
        with open(bindings_file, 'w') as f:
            f.write(bindings_code)
        
        self.installed_packages.add(package_name)
        print(f"[SUCCESS] {package_name} installed and transpiled!")
        return True
    
    def _pip_install(self, package_name: str) -> bool:
        """Install package via pip"""
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package_name
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def list_installed(self) -> List[str]:
        """List installed AquaLua packages"""
        return list(self.installed_packages)
    
    def uninstall_package(self, package_name: str) -> bool:
        """Uninstall AquaLua package"""
        bindings_file = os.path.join(self.bindings_dir, f"{package_name}.aq")
        try:
            if os.path.exists(bindings_file):
                os.remove(bindings_file)
            self.installed_packages.discard(package_name)
            print(f"[SUCCESS] Uninstalled {package_name}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to uninstall {package_name}: {e}")
            return False

# CLI Interface
def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python aqualua_transpiler.py <command> [args]")
        print("Commands:")
        print("  install <package>   - Install and transpile Python package")
        print("  list               - List installed packages")
        print("  uninstall <package> - Uninstall package")
        print("  transpile <library> - Transpile library without installing")
        return
    
    pm = AquaLuaPackageManager()
    command = sys.argv[1]
    
    if command == "install" and len(sys.argv) > 2:
        package = sys.argv[2]
        pm.install_package(package)
    
    elif command == "list":
        packages = pm.list_installed()
        print("Installed AquaLua packages:")
        for pkg in packages:
            print(f"  - {pkg}")
    
    elif command == "uninstall" and len(sys.argv) > 2:
        package = sys.argv[2]
        pm.uninstall_package(package)
    
    elif command == "transpile" and len(sys.argv) > 2:
        library = sys.argv[2]
        transpiler = LibraryTranspiler()
        code = transpiler.transpile_library(library)
        print("Generated AquaLua bindings:")
        print(code)
    
    else:
        print("Invalid command or missing arguments")

if __name__ == "__main__":
    main()