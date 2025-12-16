# Aqualua - The AI-First Programming Language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/Version-0.1_Draft-blue.svg)](https://github.com/aqualua-lang)

**The world's first statically-typed, compiled programming language designed specifically for AI and machine learning workflows.**

```aqualua
model MLP {
    l1: Linear(784, 256)
    l2: Linear(256, 10)

    fn forward(x: Tensor[B, 784]) -> Tensor[B, 10] {
        return l2(relu(l1(x)))
    }
}

dataset d = MNIST("data").batch(32).shuffle()
model m = MLP()

train m using Adam(lr=1e-3) on d for 5 epochs:
    step(batch b) {
        let out = m(b.x)
        let loss = cross_entropy(out, b.y)
        optimize(loss)
        log(loss)
    }
```

## 🚀 Overview

Aqualua combines the best of multiple worlds:
- **Python-like syntax** for ease of use
- **Rust-like type system** for safety and performance  
- **Built-in ML primitives** (tensors, models, optimizers)
- **Compile-time shape checking** to catch errors early
- **Automatic differentiation** built into the compiler

## 🎯 Key Features

### 🔥 AI-First Design
- **Tensor types**: `Tensor[32, 128, f32]` with compile-time shape validation
- **Model declarations**: First-class neural network definitions
- **Training DSL**: Natural syntax for training loops
- **Automatic differentiation**: No manual gradient computation needed

### ⚡ Performance & Safety
- **Static typing**: Catch errors at compile time, not runtime
- **Zero-cost abstractions**: High-level code compiles to optimized kernels
- **Memory safety**: Linear ownership model without garbage collection
- **Multi-backend**: CPU, GPU (CUDA), and ONNX export

### 🛠️ Developer Experience
- **Shape inference**: Smart type inference for tensors
- **Clear error messages**: Helpful diagnostics with suggestions
- **REPL support**: Interactive development and debugging
- **Framework integration**: Works with existing ML ecosystems

## 📝 Language Design Rationale

### Why Static Typing for ML?
Traditional ML frameworks use dynamic typing, leading to runtime shape errors that are expensive to debug. Aqualua's static type system catches these issues at compile time:

```aqualua
let x: Tensor[32, 784] = random(32, 784)
let w: Tensor[784, 256] = random(784, 256)
let y = x @ w  // Compiler validates: [32,784] @ [784,256] = [32,256] ✓
```

### Why the Training DSL?
Most ML code follows the same pattern: load data, define model, train in loops. Aqualua makes this a first-class language construct:

```aqualua
// Instead of writing boilerplate training loops...
train model using optimizer on dataset for epochs:
    step(batch) {
        // Your training logic here
    }
```

This enables the compiler to:
- Generate optimized training loops
- Handle device placement automatically  
- Implement gradient accumulation and mixed precision
- Add profiling and debugging hooks

### Why Control Flow Constructs?

**If Statements**: Essential for conditional training logic
```aqualua
if loss < best_loss {
    save_model(model, "best.aq")
    best_loss = loss
}
```

**While Loops**: Needed for convergence-based training
```aqualua
while loss > threshold {
    let batch = dataset.next()
    loss = train_step(batch)
}
```

**For Loops**: Critical for dataset iteration and multi-epoch training
```aqualua
for epoch in range(100) {
    for batch in dataset {
        train_step(batch)
    }
}
```

## 🏗️ Architecture

### Compiler Pipeline
```
Source (.aq) → Lexer → Parser → AST → Typechecker → Graph Extractor → IR → Optimizer → Codegen
```

**Key Components:**
- **Frontend**: Lexical analysis, parsing, and type checking
- **Graph Extractor**: Identifies differentiable computation graphs
- **IR Lowering**: Converts to MLIR for optimization
- **Backend**: Generates CUDA kernels, LLVM IR, or ONNX graphs

### Runtime System
- **Device Manager**: Automatic GPU/CPU placement and memory management
- **Tensor Runtime**: Efficient tensor operations with buffer reuse
- **Training Engine**: Optimized training loops with automatic differentiation

## 📚 Language Reference

### Types
```aqualua
// Primitives
let x: i32 = 42
let y: f32 = 3.14
let flag: bool = true

// Tensors (the core type)
let matrix: Tensor[32, 64, f32] = random(32, 64)
let image: Tensor[B, 3, 224, 224, f16] = load_batch()

// Models
model Transformer {
    embed: Embedding(vocab_size, d_model)
    layers: Sequential(TransformerBlock(), 12)
    
    fn forward(x: Tensor[B, T]) -> Tensor[B, T, vocab_size] {
        return layers(embed(x))
    }
}
```

### Built-in Operations
```aqualua
// Tensor operations
let c = a @ b           // Matrix multiplication
let d = relu(c)         // Activation functions
let e = c.sum(dim=1)    // Reductions
let f = reshape(e, [32, -1])  // Shape manipulation

// Neural network layers
Linear(in_features, out_features)
Conv2D(in_channels, out_channels, kernel_size)
BatchNorm(num_features)
Dropout(rate)
```

### Training
```aqualua
// Dataset loading
dataset train_data = ImageFolder("data/train")
    .map(resize(224, 224))
    .batch(64)
    .shuffle(1000)

// Model training
train model using Adam(lr=1e-3, weight_decay=1e-4) 
on train_data for 100 epochs:
    
    step(batch b) {
        let predictions = model(b.images)
        let loss = cross_entropy(predictions, b.labels)
        optimize(loss)
        
        if step % 100 == 0 {
            log("Loss: {}", loss)
        }
    }
```

## 🔧 Installation & Usage

### Prerequisites
- CUDA 11.8+ (for GPU support)
- Python 3.8+ (for interop)
- LLVM 15+ (for CPU codegen)

### Building from Source
```bash
git clone https://github.com/aqualua-lang/aqualua
cd aqualua
cargo build --release
```

### Usage
```bash
# Run a script
aqualua run model.aq

# Interactive REPL
aqualua repl

# Compile to executable
aqualua build model.aq -o model

# Format code
aqualua fmt src/
```

## 🎨 Examples

### Image Classification
```aqualua
model ResNet18 {
    conv1: Conv2D(3, 64, 7, stride=2, padding=3)
    bn1: BatchNorm(64)
    layer1: ResidualBlock(64, 64, 2)
    layer2: ResidualBlock(64, 128, 2)
    layer3: ResidualBlock(128, 256, 2)
    layer4: ResidualBlock(256, 512, 2)
    fc: Linear(512, 1000)
    
    fn forward(x: Tensor[B, 3, 224, 224]) -> Tensor[B, 1000] {
        let x = relu(bn1(conv1(x)))
        let x = layer1(x)
        let x = layer2(x)  
        let x = layer3(x)
        let x = layer4(x)
        let x = adaptive_avg_pool2d(x, [1, 1])
        let x = flatten(x)
        return fc(x)
    }
}
```

### Natural Language Processing
```aqualua
model GPT {
    embed: Embedding(vocab_size, d_model)
    pos_embed: PositionalEncoding(max_len, d_model)
    layers: Sequential(TransformerBlock(d_model, n_heads), n_layers)
    ln_f: LayerNorm(d_model)
    head: Linear(d_model, vocab_size)
    
    fn forward(x: Tensor[B, T]) -> Tensor[B, T, vocab_size] {
        let x = embed(x) + pos_embed(x)
        let x = layers(x)
        let x = ln_f(x)
        return head(x)
    }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `aqualua test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Credits

**Created by:** Abdullah Zeyad Alshaikh  
**Institution:** Al-ameen Middle School, Al-izdihar  
**Version:** 0.1 Draft

## 🔮 Roadmap

### v0.2 (Planned)
- [ ] GPU kernel programming (Triton-like syntax)
- [ ] Distributed training support
- [ ] ONNX model import/export
- [ ] WebGPU backend for browser deployment
- [ ] Package manager and module system

### v0.3 (Future)
- [ ] JIT compilation mode
- [ ] Quantization and pruning tools
- [ ] Real-time inference pipelines
- [ ] Visual model debugging tools

## 📞 Support

- **Documentation**: [aqualua-lang.org/docs](https://aqualua-lang.org/docs)
- **Community**: [Discord](https://discord.gg/aqualua)
- **Issues**: [GitHub Issues](https://github.com/aqualua-lang/aqualua/issues)

---

**"Making AI development as natural as writing English, as safe as Rust, and as fast as hand-optimized CUDA."**

*Built with ❤️ for the AI community*