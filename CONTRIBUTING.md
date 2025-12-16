# Contributing to AquaLua

Thank you for your interest in contributing to AquaLua! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/aqualua.git
   cd aqualua
   ```
3. **Set up development environment**:
   ```bash
   # Install Python dependencies
   pip install pyinstaller numpy
   
   # Build C runtime (Windows with Visual Studio)
   build_c_runtime_x64.bat
   ```

## 🛠️ Development Workflow

### Making Changes
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the coding standards
3. **Test your changes**:
   ```bash
   python aqualua_cli.py examples/simple_test.aq
   ```
4. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

### Testing
- Test basic functionality: `python aqualua_cli.py examples/simple_test.aq`
- Test IDE: `python aqualua_ide.py`
- Test C backend: Ensure "Using high-performance C backend" appears
- Test examples in `examples/` folder

## 📝 Coding Standards

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for public functions
- Keep functions focused and small

### C Code
- Use consistent indentation (4 spaces)
- Add comments for complex algorithms
- Follow existing naming conventions
- Ensure memory safety

### AquaLua Language
- Use consistent syntax in examples
- Add comments explaining complex concepts
- Follow the established language patterns

## 🐛 Bug Reports

When reporting bugs, please include:
- **Operating System** and version
- **Python version** (`python --version`)
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Error messages** (full stack trace if available)
- **Sample code** that demonstrates the issue

## 💡 Feature Requests

For new features:
- **Describe the use case** and motivation
- **Provide examples** of how it would be used
- **Consider backwards compatibility**
- **Discuss implementation approach** if you have ideas

## 🔧 Areas for Contribution

### High Priority
- **Performance optimizations** in C runtime
- **Additional AI/ML operations** (convolutions, attention, etc.)
- **Better error messages** and debugging support
- **Cross-platform support** (Linux, macOS)

### Medium Priority
- **Language features** (modules, packages, imports)
- **IDE improvements** (autocomplete, debugging, themes)
- **Documentation** and tutorials
- **Example programs** and use cases

### Low Priority
- **Syntax extensions** and language enhancements
- **Integration** with other tools and libraries
- **Performance benchmarks** and profiling

## 📚 Documentation

When adding features:
- Update relevant `.md` files in `distribution/`
- Add examples to `examples/` folder
- Update API documentation if adding built-ins
- Include usage examples in docstrings

## 🧪 Pull Request Process

1. **Ensure tests pass** and examples work
2. **Update documentation** for any new features
3. **Follow commit message format**:
   - `Add: new feature or functionality`
   - `Fix: bug fixes`
   - `Update: improvements to existing features`
   - `Docs: documentation changes`
4. **Submit pull request** with clear description
5. **Respond to feedback** and make requested changes

## 🏗️ Architecture Guidelines

### Adding Language Features
1. **Lexer**: Add tokens in `aqualua_lexer.py`
2. **Parser**: Add grammar rules in `aqualua_parser.py`
3. **AST**: Add node types in `aqualua_ast.py`
4. **Interpreter**: Add execution logic in `aqualua_interpreter.py`

### Adding Built-in Functions
1. **Python**: Add to `aqualua_interpreter.py`
2. **C Backend**: Add to `aqualua_runtime.c`
3. **FFI**: Add bindings in `aqualua_backend.py`
4. **Documentation**: Update `distribution/API.md`

### Performance Considerations
- **C backend first** for math-heavy operations
- **Python fallback** for compatibility
- **Memory management** - avoid leaks in C code
- **Error handling** - graceful degradation

## 🤝 Community Guidelines

- **Be respectful** and inclusive
- **Help newcomers** learn the codebase
- **Share knowledge** and best practices
- **Focus on constructive feedback**
- **Celebrate contributions** of all sizes

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Code Review**: Ask for feedback on complex changes

Thank you for contributing to AquaLua! 🚀