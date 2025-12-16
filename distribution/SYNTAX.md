# AquaLua Syntax Reference

## 🎯 Language Overview

AquaLua is an AI-first programming language with Python-like syntax, Rust-like type system, and C/CUDA backend for performance.

## 📝 Basic Syntax

### Variables and Constants
```aqualua
// Variables
let name = "AquaLua"
let age = 25
let pi = 3.14159

// Constants
const MAX_SIZE = 1000
const API_KEY = "your-key-here"

// Type annotations (optional)
let count: int = 42
let message: string = "Hello"
let active: bool = true
```

### Functions
```aqualua
// Basic function
fn greet(name: string) {
    print("Hello, " + name + "!")
}

// Function with return type
fn add(a: int, b: int) -> int {
    return a + b
}

// Colon syntax (alternative)
fn multiply(x: int, y: int): int {
    return x * y
}

// Main function
fn main() {
    greet("World")
    let result = add(5, 3)
    print(result)
}
```

### Control Flow

#### If Statements
```aqualua
if age >= 18 {
    print("Adult")
} else if age >= 13 {
    print("Teenager")
} else {
    print("Child")
}
```

#### Loops
```aqualua
// While loop
let i = 0
while i < 10 {
    print(i)
    i = i + 1
}

// For loop
for i in range(0, 10) {
    print(i)
}

// Break and continue
for i in range(0, 20) {
    if i == 5 {
        continue
    }
    if i == 15 {
        break
    }
    print(i)
}
```

### Data Types

#### Primitives
```aqualua
let integer: int = 42
let decimal: float = 3.14
let text: string = "Hello"
let flag: bool = true
let nothing: null = null
```

#### Collections
```aqualua
// Arrays
let numbers = [1, 2, 3, 4, 5]
let names = ["Alice", "Bob", "Charlie"]

// Access elements
print(numbers[0])  // 1
print(names[1])    // "Bob"
```

### Classes and Objects
```aqualua
class Person {
    fn init(name: string, age: int) {
        this.name = name
        this.age = age
    }
    
    fn greet() {
        print("Hi, I'm " + this.name)
    }
    
    fn get_age() -> int {
        return this.age
    }
}

// Create instance
let person = Person("Alice", 30)
person.greet()
print(person.get_age())
```

## 🤖 AI/ML Features

### Tensors
```aqualua
// Create tensors
tensor weights = random([10, 5])
tensor biases = zeros([5])
tensor input_data = ones([1, 10])

// Tensor operations
tensor result = matmul(weights, input_data)
tensor output = add(result, biases)

// Activation functions
tensor activated = relu(output)
tensor probabilities = softmax(activated)
```

### Neural Networks
```aqualua
// Create neural network layers
layer dense1 = dense(input_size=784, output_size=128, activation="relu")
layer dense2 = dense(input_size=128, output_size=10, activation="softmax")

// Forward pass
tensor hidden = dense1.forward(input_data)
tensor predictions = dense2.forward(hidden)
```

### Built-in AI Functions
```aqualua
// Math operations
let result = sqrt(16)        // 4.0
let power = pow(2, 8)        // 256
let maximum = max([1,5,3])   // 5

// Random operations
tensor random_tensor = random([3, 3])
let random_int = randint(1, 100)
let random_float = randf(0.0, 1.0)

// Array operations
let sum_result = sum([1, 2, 3, 4])     // 10
let mean_result = mean([2, 4, 6, 8])   // 5.0
```

## 🔧 Advanced Features

### Error Handling
```aqualua
try {
    let result = divide(10, 0)
    print(result)
} catch error {
    print("Error occurred: " + error)
} finally {
    print("Cleanup code here")
}
```

### Python Integration
```aqualua
// Execute Python code
ast_exec("
import numpy as np
result = np.array([1, 2, 3]) * 2
print(result)
")

// Use Python libraries
ast_exec("
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.show()
")
```

### Imports and Modules
```aqualua
import math
import "custom_module.aq"

fn calculate_circle_area(radius: float) -> float {
    return math.pi * pow(radius, 2)
}
```

## 📋 Operators

### Arithmetic
```aqualua
let a = 10 + 5    // Addition: 15
let b = 10 - 5    // Subtraction: 5
let c = 10 * 5    // Multiplication: 50
let d = 10 / 5    // Division: 2
let e = 10 % 3    // Modulo: 1
```

### Comparison
```aqualua
let equal = (5 == 5)        // true
let not_equal = (5 != 3)    // true
let greater = (10 > 5)      // true
let less = (3 < 8)          // true
let gte = (5 >= 5)          // true
let lte = (3 <= 7)          // true
```

### Logical
```aqualua
let and_result = true && false   // false
let or_result = true || false    // true
let not_result = !true           // false
```

## 💡 Best Practices

1. **Use meaningful variable names**
   ```aqualua
   // Good
   let user_count = 150
   
   // Bad
   let x = 150
   ```

2. **Add type annotations for clarity**
   ```aqualua
   fn process_data(data: tensor, learning_rate: float) -> tensor {
       // Function body
   }
   ```

3. **Use constants for magic numbers**
   ```aqualua
   const LEARNING_RATE = 0.001
   const BATCH_SIZE = 32
   ```

4. **Handle errors gracefully**
   ```aqualua
   try {
       let result = risky_operation()
   } catch error {
       print("Operation failed: " + error)
   }
   ```

## 🚀 Performance Tips

- Use C backend for math-heavy operations (automatic)
- Prefer tensor operations over loops for large data
- Use appropriate data types (int vs float)
- Leverage built-in AI/ML functions

## 📚 Examples

See the `examples/` folder for complete programs demonstrating:
- Basic syntax usage
- AI/ML applications
- Game development
- Data processing
- Python integration