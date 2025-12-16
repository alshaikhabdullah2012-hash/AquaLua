import subprocess
import os
import sys
import platform

def build_runtime():
    """Build precompiled C runtime for distribution"""
    
    print("Building AquaLua C Runtime...")
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("❌ This build script is for Windows only")
        return False
    
    # Find Visual Studio tools
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
    ]
    
    vcvars = None
    for path in vs_paths:
        if os.path.exists(path):
            vcvars = path
            break
    
    if not vcvars:
        print("Visual Studio Build Tools not found")
        print("Please install Visual Studio Build Tools or Community Edition")
        return False
    
    # Build command
    build_cmd = f'''
    call "{vcvars}" && ^
    cl /LD /O2 /DNDEBUG aqualua_runtime.c /Fe:aqualua_runtime.dll
    '''
    
    try:
        result = subprocess.run(
            build_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print("Runtime built successfully!")
            print("aqualua_runtime.dll is ready for distribution")
            return True
        else:
            print("Build failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Build error: {e}")
        return False

if __name__ == "__main__":
    build_runtime()