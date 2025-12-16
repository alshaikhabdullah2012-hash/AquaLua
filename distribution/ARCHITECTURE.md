# AquaLua Architecture

## 🏗️ System Overview

AquaLua uses a multi-layered architecture designed for performance, compatibility, and ease of development.

```
┌─────────────────────────────────────────┐
│              User Code (.aq)            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            AquaLua CLI/IDE              │
│         (aqualua_cli.py)                │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Lexer & Parser                 │
│    (aqualua_lexer.py, aqualua_parser.py)│
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         AST & Interpreter               │
│      (aqualua_interpreter.py)           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Backend Layer                  │
│      (aqualua_backend.py)               │
└─────────┬───────────────┬───────────────┘
          │               │
┌─────────▼─────────┐   ┌─▼───────────────┐
│   C Runtime       │   │ Python Fallback │
│ (aqualua_runtime  │   │   (NumPy, etc)  │
│     .dll/.so)     │   │                 │
└───────────────────┘   └─────────────────┘
```

## 🔧 Core Components

### 1. Frontend (Parsing)

#### Lexer (`aqualua_lexer.py`)
- **Purpose**: Tokenizes source code into meaningful symbols
- **Input**: Raw AquaLua source code
- **Output**: Stream of tokens (keywords, identifiers, operators, literals)
- **Features**:
  - Handles both `{}` and `:` syntax styles
  - Recognizes AI/ML specific keywords
  - Supports string literals, numbers, comments

#### Parser (`aqualua_parser.py`)
- **Purpose**: Builds Abstract Syntax Tree (AST) from tokens
- **Input**: Token stream from lexer
- **Output**: AST representing program structure
- **Features**:
  - Recursive descent parser
  - Error recovery and reporting
  - Supports all AquaLua language constructs
  - Handles operator precedence

### 2. Execution Engine

#### Interpreter (`aqualua_interpreter.py`)
- **Purpose**: Executes AquaLua programs by traversing AST
- **Architecture**: Tree-walking interpreter
- **Features**:
  - Variable scoping and environments
  - Function calls and returns
  - Built-in function library
  - Error handling and stack traces
  - Integration with backend systems

#### AST Nodes (`aqualua_ast.py`)
- **Purpose**: Defines structure of Abstract Syntax Tree
- **Node Types**:
  - Expressions (binary ops, function calls, literals)
  - Statements (assignments, control flow, declarations)
  - Declarations (functions, classes, variables)

### 3. Backend Layer

#### Backend Manager (`aqualua_backend.py`)
- **Purpose**: Manages high-performance computation backends
- **Strategy**: Automatic fallback system
- **Backends**:
  1. **C Runtime** (primary) - High performance
  2. **Python/NumPy** (fallback) - Full compatibility

#### C Runtime (`aqualua_runtime.c`)
- **Purpose**: High-performance mathematical operations
- **Features**:
  - Tensor operations (BLAS integration)
  - Neural network primitives
  - Memory management
  - CUDA support (future)
- **Interface**: Python FFI via ctypes
- **Performance**: 2-5x speedup over pure Python

### 4. User Interfaces

#### Command Line Interface (`aqualua_cli.py`)
- **Purpose**: Run AquaLua programs from command line
- **Features**:
  - File execution
  - REPL mode (future)
  - Debug output
  - Error reporting

#### Integrated Development Environment (`aqualua_ide.py`)
- **Purpose**: Professional development environment
- **Features**:
  - Syntax highlighting
  - File management
  - Integrated terminal
  - Real-time execution
  - Modern UI with animations

## 🚀 Execution Flow

### 1. Source Code Processing
```
hello.aq → Lexer → Tokens → Parser → AST
```

### 2. Interpretation
```
AST → Interpreter → Function Calls → Backend Selection
```

### 3. Backend Execution
```
High-level Operation → C Runtime (fast) OR Python (compatible)
```

### 4. Result Return
```
Backend Result → Interpreter → User Output
```

## 🔄 Backend Selection Logic

```python
def execute_operation(operation, *args):
    try:
        # Try C runtime first (high performance)
        if c_runtime_available():
            return c_runtime.execute(operation, *args)
    except Exception:
        pass
    
    # Fallback to Python (full compatibility)
    return python_backend.execute(operation, *args)
```

## 📊 Performance Characteristics

| Component | Performance | Purpose |
|-----------|-------------|---------|
| Lexer/Parser | Fast | One-time cost per program |
| Interpreter | Medium | Tree-walking overhead |
| C Runtime | Very Fast | Math-heavy operations |
| Python Fallback | Slower | Compatibility and features |

## 🏛️ Design Principles

### 1. **Performance First**
- C runtime for critical operations
- Automatic backend selection
- Minimal interpretation overhead

### 2. **Compatibility Always**
- Python fallback ensures programs always run
- Cross-platform support
- Graceful degradation

### 3. **Developer Experience**
- Modern IDE with professional features
- Clear error messages
- Comprehensive tooling

### 4. **AI/ML Focus**
- Built-in tensor operations
- Neural network primitives
- Seamless Python integration

## 🔧 Extension Points

### Adding New Built-ins
1. Add function to `aqualua_interpreter.py`
2. Implement in both C runtime and Python fallback
3. Add to keyword list if needed

### Backend Extensions
1. Implement backend interface
2. Add to backend selection logic
3. Register with backend manager

### Language Features
1. Add tokens to lexer
2. Add grammar rules to parser
3. Add AST node types
4. Implement in interpreter

## 📦 Distribution Architecture

### Development Setup
```
Source Files → Python Interpreter → Direct Execution
```

### Production Distribution
```
Source Files → PyInstaller → Single EXE → End User
```

### Runtime Dependencies
- **Included**: C runtime DLL, Python interpreter, all libraries
- **External**: None (self-contained executables)

## 🔮 Future Architecture

### Planned Enhancements
1. **JIT Compilation** - Compile hot paths to machine code
2. **CUDA Backend** - GPU acceleration for AI/ML workloads
3. **LLVM Backend** - Native code generation
4. **Package System** - Module management and distribution
5. **Debugger Integration** - Step-through debugging support

### Scalability Considerations
- Modular backend system allows easy addition of new runtimes
- AST-based design enables multiple execution strategies
- Clean separation of concerns supports parallel development

This architecture provides the foundation for a high-performance, user-friendly AI-first programming language that can grow and evolve with user needs.