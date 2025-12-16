# AquaLua Distribution Builder

This folder contains everything needed to build professional AquaLua executables and installer packages.

## 🚀 Quick Build

```bash
# Build everything at once
build_all.bat

# Or step by step:
python build_runtime.py      # Build C runtime DLL
python build_exe.py          # Build EXE files
python build_installer.py    # Create installer package
```

## 📦 What Gets Built

### Executables
- **aqualua.exe** - Command line interpreter with AquaLua logo
- **aqualua-ide.exe** - Professional IDE with AquaLua logo (windowed app)

### Runtime
- **aqualua_runtime.dll** - High-performance C backend
- **Python fallback** - Full compatibility when C runtime unavailable

### Distribution Package
```
AquaLua_Installer/
├── aqualua.exe              # CLI executable
├── aqualua-ide.exe          # IDE executable  
├── aqualua_runtime.dll      # C performance runtime
├── install.bat              # Auto-installer script
├── AquaLua logo.png         # Logo file
├── examples/                # Sample AquaLua programs
├── docs/                    # Complete documentation
└── README.md               # User installation guide
```

## 🛠️ Build Requirements

- **Python 3.7+**
- **PyInstaller** (auto-installed)
- **Visual Studio Build Tools** (for C runtime)
- **Windows 10/11**

## 📋 Build Scripts

| Script | Purpose |
|--------|---------|
| `build_all.bat` | Master build script - runs everything |
| `build_runtime.py` | Compiles C runtime to DLL |
| `build_exe.py` | Creates EXE files with PyInstaller |
| `build_installer.py` | Packages complete installer |

## 🎯 Distribution Features

- ✅ **Single-file executables** - No dependencies
- ✅ **Custom AquaLua logo** - Professional branding
- ✅ **Auto-installer** - One-click setup
- ✅ **Complete documentation** - All guides included
- ✅ **Example programs** - Ready-to-run samples
- ✅ **High performance** - C runtime included

## 📁 Output Structure

After building, you'll have:
- `dist/` - Individual EXE files
- `AquaLua_Installer/` - Complete distribution package
- `build/` - Temporary build files

## 🚀 User Experience

1. User downloads `AquaLua_Installer.zip`
2. Extracts and runs `install.bat` as Administrator
3. AquaLua installed to `C:\AquaLua` and added to PATH
4. Can immediately use `aqualua` and `aqualua-ide` commands

## 📖 Documentation Included

- Installation guide
- Language syntax reference
- Architecture overview
- API documentation
- Example programs
- Troubleshooting guide

Perfect for professional software distribution!