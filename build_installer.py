import os
import shutil
import subprocess
import sys

def create_installer():
    """Create complete AquaLua installer package"""
    
    print("Creating AquaLua Installer Package...")
    
    # Create installer directory
    installer_dir = "AquaLua_Installer"
    if os.path.exists(installer_dir):
        shutil.rmtree(installer_dir)
    os.makedirs(installer_dir)
    
    # Build executables first
    print("Building executables...")
    subprocess.run([sys.executable, "build_exe.py"], check=True)
    
    # Copy executables
    dist_dir = "dist"
    if os.path.exists(dist_dir):
        for exe_file in os.listdir(dist_dir):
            if exe_file.endswith(".exe"):
                shutil.copy2(
                    os.path.join(dist_dir, exe_file),
                    os.path.join(installer_dir, exe_file)
                )
                print(f"Copied {exe_file}")
    
    # Copy runtime files
    runtime_files = [
        "aqualua_runtime.dll",
        "aqualua_backend.py",
        "aqualua_interpreter.py",
        "aqualua_parser.py",
        "aqualua_lexer.py"
    ]
    
    for file in runtime_files:
        if os.path.exists(file):
            shutil.copy2(file, installer_dir)
            print(f"Copied {file}")
    
    # Copy documentation
    docs_dir = os.path.join(installer_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    doc_files = [
        "distribution/README.md",
        "distribution/SYNTAX.md", 
        "distribution/ARCHITECTURE.md",
        "distribution/API.md",
        "distribution/INSTALLATION.md",
        "distribution/EXAMPLES.md",
        "distribution/TROUBLESHOOTING.md"
    ]
    
    for file in doc_files:
        if os.path.exists(file):
            dest_name = os.path.basename(file)
            shutil.copy2(file, os.path.join(docs_dir, dest_name))
            print(f"Copied {dest_name} to docs/")
    
    # Copy logo
    logo_src = r"C:\Users\abood\Downloads\Aqualua\AquaLua logo\AquaLua logo.png"
    if os.path.exists(logo_src):
        shutil.copy2(logo_src, os.path.join(installer_dir, "AquaLua logo.png"))
        print("Copied logo")
    
    # Copy examples
    if os.path.exists("examples"):
        shutil.copytree("examples", os.path.join(installer_dir, "examples"))
        print("Copied examples")
    
    # Create installer script
    installer_script = f'''@echo off
echo Installing AquaLua...

:: Create AquaLua directory
if not exist "C:\\AquaLua" mkdir "C:\\AquaLua"

:: Copy files
xcopy /Y /E "{os.path.abspath(installer_dir)}\\*" "C:\\AquaLua\\"

:: Add to PATH
setx PATH "%PATH%;C:\\AquaLua" /M

echo AquaLua installed successfully!
echo Installed to: C:\\AquaLua
echo Usage:
echo    aqualua file.aq    - Run AquaLua file
echo    aqualua-ide        - Launch IDE
echo.
echo Press any key to exit...
pause >nul
'''
    
    with open(os.path.join(installer_dir, "install.bat"), "w") as f:
        f.write(installer_script)
    
    # Create README
    readme_content = '''# AquaLua - AI-First Programming Language

## Installation

1. Run `install.bat` as Administrator
2. AquaLua will be installed to C:\\AquaLua
3. Executables will be added to PATH

## Usage

### Command Line
```
aqualua hello.aq
```

### IDE
```
aqualua-ide
```

## What's Included

- aqualua.exe - Command line interpreter
- aqualua-ide.exe - Professional IDE
- C Runtime (aqualua_runtime.dll) - High performance
- Python Fallback - Full compatibility
- Examples and Documentation
- AI/ML Built-ins

## Requirements

- Windows 10/11
- No additional dependencies needed!

## Features

- High-performance C backend
- Modern IDE with syntax highlighting
- Built-in AI/ML operations
- Ready-to-run executables
- No compilation required

Enjoy coding with AquaLua!
'''
    
    with open(os.path.join(installer_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"\nInstaller package created!")
    print(f"Location: {os.path.abspath(installer_dir)}")
    print("Contents:")
    print("   - aqualua.exe")
    print("   - aqualua-ide.exe") 
    print("   - Runtime files")
    print("   - install.bat")
    print("   - Examples")
    print("   - Documentation")

if __name__ == "__main__":
    create_installer()