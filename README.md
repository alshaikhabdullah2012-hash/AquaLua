# AquaLua Programming Language

**AI-First Programming Language with High-Performance C Backend**

AquaLua is a modern programming language designed specifically for AI and machine learning development, featuring Python-like syntax, Rust-like type system, and a high-performance C/CUDA backend.

## 🚀 Features

- **High Performance**: C runtime backend with 2-5x speedup over pure Python
- **AI/ML First**: Built-in tensor operations, neural networks, and ML primitives
- **Python Integration**: Seamless interop with Python libraries and ecosystem
- **Modern Syntax**: Clean, readable syntax inspired by Python and Rust
- **Professional IDE**: Full-featured development environment with syntax highlighting
- **Zero Dependencies**: Precompiled executables require no additional installations

## 📦 Quick Start

### Download & Install
1. Download the latest release from [Releases](../../releases) Note:if you are early there may not be any releases
2. Extract `AquaLua_Installer.zip`
3. Run `install.bat` as Administrator
4. Start coding with `aqualua` and `aqualua-ide`

### Hello World
```
print("Hello world")
#if you want to write it in python
ast_exec("
import math # this is just an example that you can import anything
print("hello world")
")
```

### Run Your Program
```bash
aqualua hello.aq        # Command line
aqualua-ide            # Launch IDE
```

## 🧠 AI/ML Capabilities

### Tensor Operations
```aqualua
tensor a = random([100, 50])
tensor b = ones([50, 25])
tensor result = matmul(a, b)
tensor activated = relu(result)
```

### Neural Networks
```aqualua
layer dense1 = dense(784, 128, "relu")
layer output = dense(128, 10, "softmax")

tensor predictions = output.forward(dense1.forward(input_data))
```

### Python Integration
```aqualua
ast_exec("
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)
plt.hist(data, bins=50)
plt.show()
")
```

## 🏗️ Architecture

- **Frontend**: Lexer, Parser, AST (Python)
- **Backend**: High-performance C runtime with automatic fallback
- **Runtime**: Dual-backend system (C for speed, Python for compatibility)
- **IDE**: Modern tkinter-based development environment

## 📚 Documentation

- [Language Syntax](distribution/SYNTAX.md) - Complete syntax reference
- [API Reference](distribution/API.md) - Built-in functions and operations
- [Architecture](distribution/ARCHITECTURE.md) - System design and components
- [Installation Guide](distribution/INSTALLATION.md) - Detailed setup instructions
- [Examples](distribution/EXAMPLES.md) - Code examples and tutorials
- [Troubleshooting](distribution/TROUBLESHOOTING.md) - Common issues and solutions

## 🛠️ Development

### Building from Source
```bash
# Build C runtime (requires Visual Studio)
build_c_runtime_x64.bat

# Build executables
python build_exe.py

# Create installer package
python build_installer.py
```

### Requirements
- Python 3.7+
- Visual Studio Build Tools (for C runtime)
- PyInstaller (auto-installed)

## 🎯 Performance

| Operation | Python | AquaLua C Backend | Speedup |
|-----------|--------|-------------------|---------|
| Matrix Multiplication | 100ms | 45ms | 2.2x |
| Tensor Operations | 80ms | 32ms | 2.5x |
| Neural Network Forward | 150ms | 65ms | 2.3x |

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 🔗 Links

- [Documentation](distribution/)
- [Examples](examples/)
- [Issues](../../issues)
- [Releases](../../releases)

---

**AquaLua** - Making AI development faster and more intuitive."# AquaLua" 

