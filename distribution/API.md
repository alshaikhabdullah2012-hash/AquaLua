# AquaLua API Reference

## 🔧 Built-in Functions

### Basic I/O

#### `print(value)`
Outputs value to console.
```aqualua
print("Hello, World!")
print(42)
print(true)
```

#### `input(prompt)`
Gets user input from console.
```aqualua
let name = input("Enter your name: ")
let age = int(input("Enter your age: "))
```

### Type Conversion

#### `int(value)`
Converts value to integer.
```aqualua
let num = int("42")        // 42
let rounded = int(3.14)    // 3
```

#### `float(value)`
Converts value to floating-point number.
```aqualua
let pi = float("3.14159")  // 3.14159
let decimal = float(42)    // 42.0
```

#### `string(value)`
Converts value to string.
```aqualua
let text = string(42)      // "42"
let flag_text = string(true) // "true"
```

#### `bool(value)`
Converts value to boolean.
```aqualua
let flag = bool(1)         // true
let empty = bool(0)        // false
let text_bool = bool("hello") // true
```

## 🧮 Mathematical Functions

### Basic Math

#### `abs(x)`
Returns absolute value.
```aqualua
let positive = abs(-5)     // 5
let same = abs(3)          // 3
```

#### `sqrt(x)`
Returns square root.
```aqualua
let root = sqrt(16)        // 4.0
let pi_root = sqrt(9.86)   // 3.14...
```

#### `pow(base, exponent)`
Returns base raised to exponent.
```aqualua
let squared = pow(5, 2)    // 25
let cubed = pow(2, 3)      // 8
```

#### `min(array)` / `max(array)`
Returns minimum/maximum value from array.
```aqualua
let smallest = min([3, 1, 4, 1, 5])  // 1
let largest = max([3, 1, 4, 1, 5])   // 5
```

#### `sum(array)`
Returns sum of all values in array.
```aqualua
let total = sum([1, 2, 3, 4, 5])     // 15
```

#### `mean(array)`
Returns average of all values in array.
```aqualua
let average = mean([2, 4, 6, 8])     // 5.0
```

### Trigonometric Functions

#### `sin(x)` / `cos(x)` / `tan(x)`
Trigonometric functions (x in radians).
```aqualua
let sine = sin(1.5708)     // ~1.0 (π/2)
let cosine = cos(0)        // 1.0
let tangent = tan(0.7854)  // ~1.0 (π/4)
```

## 🎲 Random Functions

#### `random(shape)`
Creates tensor with random values between 0 and 1.
```aqualua
tensor rand_matrix = random([3, 3])
tensor rand_vector = random([5])
```

#### `randint(min, max)`
Returns random integer between min and max (inclusive).
```aqualua
let dice = randint(1, 6)
let lottery = randint(1, 100)
```

#### `randf(min, max)`
Returns random float between min and max.
```aqualua
let probability = randf(0.0, 1.0)
let temperature = randf(-10.0, 35.0)
```

## 🧠 Tensor Operations

### Tensor Creation

#### `zeros(shape)`
Creates tensor filled with zeros.
```aqualua
tensor zero_matrix = zeros([3, 3])
tensor zero_vector = zeros([10])
```

#### `ones(shape)`
Creates tensor filled with ones.
```aqualua
tensor one_matrix = ones([2, 4])
tensor one_vector = ones([5])
```

#### `eye(size)`
Creates identity matrix.
```aqualua
tensor identity = eye(3)  // 3x3 identity matrix
```

### Tensor Operations

#### `matmul(a, b)`
Matrix multiplication.
```aqualua
tensor weights = random([10, 5])
tensor input_data = ones([5, 1])
tensor result = matmul(weights, input_data)
```

#### `add(a, b)` / `sub(a, b)` / `mul(a, b)` / `div(a, b)`
Element-wise tensor operations.
```aqualua
tensor a = ones([3, 3])
tensor b = random([3, 3])

tensor sum_result = add(a, b)
tensor diff = sub(a, b)
tensor product = mul(a, b)
tensor quotient = div(a, b)
```

#### `transpose(tensor)`
Transposes a tensor.
```aqualua
tensor matrix = random([3, 4])
tensor transposed = transpose(matrix)  // Shape: [4, 3]
```

#### `reshape(tensor, new_shape)`
Reshapes tensor to new dimensions.
```aqualua
tensor vector = ones([12])
tensor matrix = reshape(vector, [3, 4])
```

## 🤖 Neural Network Functions

### Activation Functions

#### `relu(x)`
Rectified Linear Unit activation.
```aqualua
tensor activated = relu(input_tensor)
```

#### `sigmoid(x)`
Sigmoid activation function.
```aqualua
tensor probabilities = sigmoid(logits)
```

#### `tanh(x)`
Hyperbolic tangent activation.
```aqualua
tensor normalized = tanh(input_data)
```

#### `softmax(x)`
Softmax activation for classification.
```aqualua
tensor class_probs = softmax(output_logits)
```

### Loss Functions

#### `mse_loss(predictions, targets)`
Mean Squared Error loss.
```aqualua
tensor loss = mse_loss(model_output, ground_truth)
```

#### `cross_entropy_loss(predictions, targets)`
Cross-entropy loss for classification.
```aqualua
tensor loss = cross_entropy_loss(class_probs, true_labels)
```

### Layer Operations

#### `dense(input_size, output_size, activation)`
Creates a dense (fully connected) layer.
```aqualua
layer fc1 = dense(784, 128, "relu")
layer fc2 = dense(128, 10, "softmax")

// Forward pass
tensor hidden = fc1.forward(input_data)
tensor output = fc2.forward(hidden)
```

#### `conv2d(filters, kernel_size, activation)`
Creates a 2D convolutional layer.
```aqualua
layer conv = conv2d(32, [3, 3], "relu")
tensor feature_maps = conv.forward(image_data)
```

## 🐍 Python Integration

#### `ast_exec(python_code)`
Executes Python code within AquaLua.
```aqualua
ast_exec("
import numpy as np
import matplotlib.pyplot as plt

# Create and plot data
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sine Wave')
plt.show()
")
```

#### Advanced Python Integration
```aqualua
// Use any Python library
ast_exec("
import pandas as pd
import seaborn as sns

# Load and analyze data
df = pd.read_csv('data.csv')
sns.heatmap(df.corr())
")

// Machine learning with scikit-learn
ast_exec("
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
")
```

## 🔧 Utility Functions

### Array Operations

#### `len(array)`
Returns length of array.
```aqualua
let numbers = [1, 2, 3, 4, 5]
let count = len(numbers)  // 5
```

#### `range(start, end, step)`
Creates array of numbers in range.
```aqualua
let numbers = range(0, 10, 1)    // [0, 1, 2, ..., 9]
let evens = range(0, 20, 2)      // [0, 2, 4, ..., 18]
```

#### `sort(array)`
Returns sorted copy of array.
```aqualua
let unsorted = [3, 1, 4, 1, 5]
let sorted_array = sort(unsorted)  // [1, 1, 3, 4, 5]
```

### String Operations

#### `upper(string)` / `lower(string)`
Converts string case.
```aqualua
let loud = upper("hello")     // "HELLO"
let quiet = lower("WORLD")    // "world"
```

#### `split(string, delimiter)`
Splits string into array.
```aqualua
let words = split("hello,world,test", ",")  // ["hello", "world", "test"]
```

#### `join(array, separator)`
Joins array elements into string.
```aqualua
let sentence = join(["Hello", "World"], " ")  // "Hello World"
```

## 🎮 Advanced Features

### File Operations
```aqualua
// Read file
ast_exec("
with open('data.txt', 'r') as f:
    content = f.read()
    print(content)
")

// Write file
ast_exec("
with open('output.txt', 'w') as f:
    f.write('Hello from AquaLua!')
")
```

### Web Requests
```aqualua
ast_exec("
import requests
import json

response = requests.get('https://api.github.com/users/octocat')
data = response.json()
print(f'User: {data[\"name\"]}')
")
```

### Data Visualization
```aqualua
ast_exec("
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.legend()
plt.grid(True)
plt.title('Trigonometric Functions')
plt.show()
")
```

## 🚀 Performance Notes

- **C Runtime**: Functions like `matmul`, `add`, `relu` use optimized C implementations
- **Python Fallback**: Automatically used if C runtime unavailable
- **Memory Management**: Tensors are automatically managed
- **GPU Support**: Future versions will support CUDA acceleration

## 🔍 Error Handling

All API functions include proper error handling:
```aqualua
try {
    tensor result = matmul(a, b)  // May fail if shapes incompatible
    print("Success!")
} catch error {
    print("Matrix multiplication failed: " + error)
}
```

## 📚 Examples

See the `examples/` directory for complete programs demonstrating API usage in real applications.