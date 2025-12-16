# AquaLua Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Installation Problems

#### Issue: "aqualua is not recognized as a command"
**Symptoms:**
- Command prompt shows: `'aqualua' is not recognized as an internal or external command`
- PowerShell shows: `aqualua : The term 'aqualua' is not recognized`

**Solutions:**
```bash
# 1. Check if PATH was updated
echo %PATH%
# Look for C:\AquaLua in the output

# 2. Manually add to PATH
setx PATH "%PATH%;C:\AquaLua" /M

# 3. Restart Command Prompt/PowerShell

# 4. Use full path as workaround
C:\AquaLua\aqualua.exe hello.aq
```

#### Issue: "Access denied" during installation
**Symptoms:**
- Installation fails with permission errors
- Cannot write to C:\AquaLua directory

**Solutions:**
```bash
# 1. Run install.bat as Administrator
# Right-click install.bat → "Run as Administrator"

# 2. Install to user directory instead
mkdir %USERPROFILE%\AquaLua
copy *.exe %USERPROFILE%\AquaLua\
setx PATH "%PATH%;%USERPROFILE%\AquaLua"

# 3. Check Windows Defender/Antivirus
# Add AquaLua folder to exclusions
```

### Runtime Errors

#### Issue: "DLL not found" or "C runtime failed"
**Symptoms:**
- Error: `aqualua_runtime.dll not found`
- Fallback message: `Using Python backend`

**Solutions:**
```bash
# 1. Verify DLL exists
dir C:\AquaLua\aqualua_runtime.dll

# 2. Force Python fallback (temporary fix)
set AQUALUA_FORCE_PYTHON=1
aqualua hello.aq

# 3. Reinstall C runtime
python build_runtime.py

# 4. Check Visual C++ Redistributables
# Download and install latest VC++ Redist from Microsoft
```

#### Issue: "Python not found in PATH"
**Symptoms:**
- Error when running AquaLua programs
- `python` command not recognized

**Solutions:**
```bash
# 1. Install Python 3.7+
# Download from python.org

# 2. Add Python to PATH during installation
# Check "Add Python to PATH" option

# 3. Verify Python installation
python --version
python -c "print('Python works!')"

# 4. Use specific Python version
py -3 aqualua_cli.py hello.aq
```

### IDE Issues

#### Issue: IDE won't start
**Symptoms:**
- Double-clicking aqualua-ide.exe does nothing
- Error about tkinter missing

**Solutions:**
```bash
# 1. Test tkinter availability
python -c "import tkinter; print('tkinter OK')"

# 2. If tkinter missing, reinstall Python
# Make sure to include tkinter in installation

# 3. Run IDE from command line for error details
aqualua-ide.exe

# 4. Check Windows Event Viewer for crash details
```

#### Issue: IDE crashes when running programs
**Symptoms:**
- IDE closes unexpectedly when clicking "Run"
- No output in terminal

**Solutions:**
```bash
# 1. Check if CLI works independently
aqualua hello.aq

# 2. Run IDE from command prompt to see errors
cd C:\AquaLua
aqualua-ide.exe

# 3. Increase timeout in IDE settings
# Edit aqualua_ide.py, increase timeout value

# 4. Use simpler test program
fn main() {
    print("Hello!")
}
```

### Performance Issues

#### Issue: Slow execution
**Symptoms:**
- Programs take much longer than expected
- High CPU usage

**Diagnostics:**
```bash
# Check which backend is being used
aqualua --debug hello.aq

# Should show: "Using C runtime" for best performance
# If shows: "Using Python fallback" - C runtime not working
```

**Solutions:**
```bash
# 1. Ensure C runtime is working
dir C:\AquaLua\aqualua_runtime.dll

# 2. Force C backend
set AQUALUA_BACKEND=C
aqualua hello.aq

# 3. Optimize your code
# Use tensor operations instead of loops
# Avoid nested loops with large datasets

# 4. Check system resources
# Close other applications
# Ensure sufficient RAM available
```

#### Issue: Memory errors with large tensors
**Symptoms:**
- `MemoryError` when creating large tensors
- System becomes unresponsive

**Solutions:**
```aqualua
// 1. Use smaller batch sizes
tensor large_data = random([1000, 1000])  // Instead of [10000, 10000]

// 2. Process data in chunks
for i in range(0, data_size, batch_size) {
    tensor batch = get_batch(data, i, batch_size)
    process_batch(batch)
}

// 3. Use appropriate data types
tensor float_data = random([100, 100])  // 32-bit floats
// Instead of double precision if not needed
```

### Syntax and Language Issues

#### Issue: Syntax errors in valid-looking code
**Symptoms:**
- Parser errors on seemingly correct syntax
- Unexpected token errors

**Common Causes and Fixes:**
```aqualua
// 1. Mixed brace and colon syntax
// ❌ Don't mix styles
fn test() {
    if condition:  // Colon style
        return     // But using braces above
}

// ✅ Use consistent style
fn test() {
    if condition {
        return
    }
}

// 2. Missing semicolons in some contexts
// ❌ 
let x = 5
let y = 10

// ✅ 
let x = 5
let y = 10

// 3. Incorrect function syntax
// ❌ 
function test() {  // Wrong keyword
    return
}

// ✅ 
fn test() {
    return
}
```

#### Issue: "else if" not working
**Symptoms:**
- Parser error on `else if` statements
- Unexpected token 'if'

**Solution:**
```aqualua
// ❌ Don't use 'elif'
if condition {
    // code
} elif other_condition {  // Wrong!
    // code
}

// ✅ Use 'else if' (two words)
if condition {
    // code
} else if other_condition {
    // code
}
```

### Python Integration Issues

#### Issue: Python libraries not found
**Symptoms:**
- `ModuleNotFoundError` when using `ast_exec`
- Import errors in Python code

**Solutions:**
```bash
# 1. Install missing packages
pip install numpy pandas matplotlib scikit-learn

# 2. Check Python environment
python -c "import numpy; print('NumPy version:', numpy.__version__)"

# 3. Use virtual environment
python -m venv aqualua_env
aqualua_env\Scripts\activate
pip install -r requirements.txt

# 4. Specify full Python path in code
ast_exec("
import sys
sys.path.append('C:/path/to/your/packages')
import your_module
")
```

#### Issue: Python code execution fails
**Symptoms:**
- `ast_exec` throws errors
- Python syntax errors in AquaLua

**Solutions:**
```aqualua
// 1. Check Python syntax separately
ast_exec("
print('Testing Python execution')
import sys
print('Python version:', sys.version)
")

// 2. Use proper string escaping
ast_exec("
text = 'Hello, World!'  # Use single quotes inside
print(text)
")

// 3. Handle multi-line strings carefully
ast_exec("
for i in range(5):
    print(f'Number: {i}')
")
```

## 🔧 Debugging Techniques

### Enable Debug Mode
```bash
# Get detailed execution information
aqualua --debug program.aq

# Shows:
# - Backend selection (C vs Python)
# - Parsing steps
# - Execution trace
# - Performance metrics
```

### Check System Information
```bash
# Verify installation
aqualua --version
aqualua --info

# Check dependencies
python --version
python -c "import ctypes; print('ctypes OK')"
python -c "import numpy; print('NumPy OK')"
```

### Test Components Individually
```bash
# Test lexer
python aqualua_lexer.py test.aq

# Test parser
python aqualua_parser.py test.aq

# Test interpreter
python aqualua_interpreter.py test.aq

# Test C runtime
python -c "
import ctypes
dll = ctypes.CDLL('./aqualua_runtime.dll')
print('C runtime loaded successfully')
"
```

## 📊 Performance Monitoring

### Benchmark Your Code
```aqualua
// benchmark.aq
fn benchmark_function(func_name: string, iterations: int) {
    ast_exec("
import time
start_time = time.time()
")
    
    // Your code here
    for i in range(0, iterations) {
        // Function to benchmark
    }
    
    ast_exec("
end_time = time.time()
execution_time = end_time - start_time
print(f'Function: " + func_name + "')
print(f'Iterations: " + string(iterations) + "')
print(f'Total time: {execution_time:.3f} seconds')
print(f'Time per iteration: {execution_time/" + string(iterations) + ":.6f} seconds')
")
}
```

### Memory Usage Monitoring
```aqualua
// memory_monitor.aq
fn monitor_memory() {
    ast_exec("
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f'Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB')
print(f'Virtual memory: {memory_info.vms / 1024 / 1024:.1f} MB')
")
}
```

## 🆘 Getting Help

### Built-in Help
```bash
aqualua --help          # Command line options
aqualua --version       # Version information
aqualua --debug file.aq # Debug output
```

### Documentation
- `README.md` - Quick start guide
- `SYNTAX.md` - Language reference
- `API.md` - Function documentation
- `EXAMPLES.md` - Sample programs
- `ARCHITECTURE.md` - System design

### Community Support
- GitHub Issues (if available)
- Stack Overflow (tag: aqualua)
- Community forums
- Discord/Slack channels

### Reporting Bugs

When reporting issues, include:

1. **System Information:**
   ```bash
   aqualua --version
   python --version
   # Windows version
   ```

2. **Error Messages:**
   - Full error text
   - Stack traces
   - Debug output (`--debug` flag)

3. **Minimal Reproduction:**
   - Smallest code that reproduces the issue
   - Steps to reproduce
   - Expected vs actual behavior

4. **Environment Details:**
   - Installation method
   - Antivirus software
   - Other Python installations
   - Recent system changes

### Emergency Workarounds

If AquaLua is completely broken:

```bash
# 1. Use Python directly
python aqualua_cli.py program.aq

# 2. Force Python backend
set AQUALUA_FORCE_PYTHON=1
aqualua program.aq

# 3. Reinstall from scratch
rmdir /s C:\AquaLua
# Re-run installer

# 4. Use portable installation
# Extract to different folder
# Use full paths to executables
```

## 🔄 Recovery Procedures

### Complete Reinstallation
```bash
# 1. Uninstall current version
C:\AquaLua\uninstall.bat

# 2. Clean registry (optional)
# Remove PATH entries manually

# 3. Download fresh installer
# Extract AquaLua_Installer.zip

# 4. Run installation
install.bat

# 5. Verify installation
aqualua --version
aqualua-ide
```

### Reset to Defaults
```bash
# 1. Clear environment variables
set AQUALUA_BACKEND=
set AQUALUA_FORCE_PYTHON=

# 2. Reset PATH
# Remove and re-add C:\AquaLua

# 3. Clear temporary files
del /q %TEMP%\aqualua_*

# 4. Restart system
```

Remember: Most issues can be resolved by ensuring proper installation, checking PATH variables, and verifying that all dependencies (Python, C runtime) are correctly installed and accessible.