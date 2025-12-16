# AquaLua Installation Guide

## 🚀 Quick Install (Recommended)

### For End Users
1. **Download** the `AquaLua_Installer.zip` package
2. **Extract** to any folder
3. **Run** `install.bat` as Administrator
4. **Start coding** with `aqualua` and `aqualua-ide`

### What Gets Installed
- `C:\AquaLua\` - Main installation directory
- `aqualua.exe` - Command line interpreter
- `aqualua-ide.exe` - Professional IDE
- `aqualua_runtime.dll` - High-performance C backend
- Examples and documentation
- PATH environment variable updated

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10 (64-bit) or later
- **RAM**: 2 GB minimum, 4 GB recommended
- **Storage**: 50 MB for installation
- **Display**: 1024x768 minimum resolution

### Recommended Requirements
- **OS**: Windows 11 (64-bit)
- **RAM**: 8 GB or more
- **Storage**: 1 GB free space (for projects)
- **Display**: 1920x1080 or higher
- **CPU**: Multi-core processor for better performance

## 🛠️ Installation Methods

### Method 1: Automated Installer (Easiest)
```bash
# 1. Download AquaLua_Installer.zip
# 2. Extract files
# 3. Right-click install.bat → "Run as Administrator"
# 4. Follow prompts
```

### Method 2: Manual Installation
```bash
# 1. Create directory
mkdir C:\AquaLua

# 2. Copy files
copy aqualua.exe C:\AquaLua\
copy aqualua-ide.exe C:\AquaLua\
copy aqualua_runtime.dll C:\AquaLua\

# 3. Add to PATH
setx PATH "%PATH%;C:\AquaLua" /M
```

### Method 3: Portable Installation
```bash
# 1. Extract to any folder (e.g., D:\MyApps\AquaLua)
# 2. Use full paths to executables
D:\MyApps\AquaLua\aqualua.exe hello.aq
D:\MyApps\AquaLua\aqualua-ide.exe
```

## ✅ Verify Installation

### Test Command Line
```bash
# Open Command Prompt or PowerShell
aqualua --version

# Should output: AquaLua v1.0 - AI-First Programming Language
```

### Test IDE
```bash
# Launch IDE
aqualua-ide

# Should open professional IDE with AquaLua logo
```

### Test Sample Program
Create `hello.aq`:
```aqualua
fn main() {
    print("Hello, AquaLua!")
    
    // Test AI/ML features
    tensor matrix = random([3, 3])
    print("Random matrix created successfully!")
}
```

Run it:
```bash
aqualua hello.aq
```

Expected output:
```
Hello, AquaLua!
Random matrix created successfully!
```

## 🔧 Configuration

### Environment Variables
- `AQUALUA_HOME` - Installation directory (auto-set)
- `AQUALUA_FORCE_PYTHON` - Force Python fallback (optional)
- `PATH` - Includes AquaLua executables (auto-updated)

### Performance Settings
```bash
# Force high-performance C runtime
set AQUALUA_BACKEND=C

# Force Python compatibility mode
set AQUALUA_BACKEND=PYTHON

# Auto-detect (default)
set AQUALUA_BACKEND=AUTO
```

## 🐛 Troubleshooting

### Common Issues

#### "aqualua is not recognized as a command"
**Solution**: PATH not updated correctly
```bash
# Check PATH
echo %PATH%

# Manually add if missing
setx PATH "%PATH%;C:\AquaLua" /M

# Restart Command Prompt
```

#### "DLL not found" error
**Solution**: C runtime missing or corrupted
```bash
# Force Python fallback
set AQUALUA_FORCE_PYTHON=1
aqualua hello.aq

# Or reinstall with install.bat
```

#### IDE won't start
**Solution**: Check Python/tkinter installation
```bash
# Test tkinter
python -c "import tkinter; print('OK')"

# If fails, reinstall Python with tkinter
```

#### "Access denied" during installation
**Solution**: Run as Administrator
```bash
# Right-click install.bat
# Select "Run as Administrator"
```

### Performance Issues

#### Slow execution
```bash
# Check which backend is being used
aqualua --debug hello.aq

# Should show: "Using C runtime" for best performance
# If shows "Using Python fallback", C runtime not working
```

#### Memory issues with large tensors
```bash
# Monitor memory usage
# Consider using smaller batch sizes
# Use tensor operations instead of loops
```

### Getting Help

#### Built-in Help
```bash
aqualua --help          # Command line options
aqualua --version       # Version information
aqualua --debug file.aq # Debug output
```

#### Documentation
- `README.md` - Quick start guide
- `SYNTAX.md` - Language reference
- `API.md` - Function documentation
- `examples/` - Sample programs

#### Online Resources
- GitHub repository (if available)
- Community forums
- Issue tracker

## 🔄 Updating AquaLua

### Automatic Update (Future)
```bash
aqualua --update
```

### Manual Update
1. Download new `AquaLua_Installer.zip`
2. Run `install.bat` (overwrites existing installation)
3. Existing projects and settings preserved

## 🗑️ Uninstallation

### Automated Uninstall
```bash
# Run from AquaLua directory
C:\AquaLua\uninstall.bat
```

### Manual Uninstall
```bash
# 1. Remove from PATH
# 2. Delete installation directory
rmdir /s C:\AquaLua

# 3. Clean registry (optional)
# Remove HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment\PATH entries
```

## 🚀 Advanced Installation

### Developer Installation
For those who want to modify AquaLua:

```bash
# 1. Install Python 3.7+
# 2. Clone/download source code
# 3. Install dependencies
pip install pyinstaller

# 4. Build from source
python build_all.py

# 5. Use development version
python aqualua_cli.py hello.aq
```

### Custom Installation Directory
```bash
# Install to custom location
mkdir D:\MyPrograms\AquaLua
copy *.exe D:\MyPrograms\AquaLua\
setx PATH "%PATH%;D:\MyPrograms\AquaLua" /M
```

### Network Installation
For organizations deploying to multiple machines:

```bash
# 1. Create network share
\\server\software\AquaLua\

# 2. Deploy via Group Policy or script
# 3. Update PATH on all machines
# 4. Test installation on sample machines
```

## 📊 Installation Verification Checklist

- [ ] `aqualua --version` works
- [ ] `aqualua-ide` launches successfully
- [ ] Sample program runs without errors
- [ ] C runtime loads (check with `--debug`)
- [ ] IDE shows syntax highlighting
- [ ] File operations work in IDE
- [ ] Examples folder accessible
- [ ] Documentation files present

## 🎯 Next Steps

After successful installation:

1. **Try Examples**: Explore `examples/` folder
2. **Read Documentation**: Check `SYNTAX.md` and `API.md`
3. **Create First Project**: Start with simple programs
4. **Join Community**: Connect with other AquaLua developers
5. **Build Something Amazing**: Use AI/ML features for your projects

Welcome to AquaLua! 🚀