# AquaLua Examples

## 🚀 Getting Started Examples

### Hello World
```aqualua
// hello.aq
fn main() {
    print("Hello, AquaLua!")
}
```

### Basic Variables and Functions
```aqualua
// basics.aq
fn greet(name: string) -> string {
    return "Hello, " + name + "!"
}

fn main() {
    let user_name = "Alice"
    let message = greet(user_name)
    print(message)
    
    let age = 25
    if age >= 18 {
        print("You are an adult!")
    }
}
```

### Control Flow
```aqualua
// control_flow.aq
fn main() {
    // For loop
    for i in range(1, 6) {
        print("Count: " + string(i))
    }
    
    // While loop
    let countdown = 5
    while countdown > 0 {
        print("T-minus " + string(countdown))
        countdown = countdown - 1
    }
    print("Blast off! 🚀")
}
```

## 🧮 Mathematical Examples

### Basic Math Operations
```aqualua
// math_demo.aq
fn main() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("Numbers: " + string(numbers))
    print("Sum: " + string(sum(numbers)))
    print("Average: " + string(mean(numbers)))
    print("Maximum: " + string(max(numbers)))
    print("Minimum: " + string(min(numbers)))
    
    // Advanced math
    let circle_radius = 5.0
    let area = 3.14159 * pow(circle_radius, 2)
    print("Circle area: " + string(area))
}
```

### Statistical Analysis
```aqualua
// statistics.aq
fn calculate_statistics(data: array) {
    let total = sum(data)
    let average = mean(data)
    let maximum = max(data)
    let minimum = min(data)
    
    print("Dataset Analysis:")
    print("  Count: " + string(len(data)))
    print("  Sum: " + string(total))
    print("  Mean: " + string(average))
    print("  Max: " + string(maximum))
    print("  Min: " + string(minimum))
    print("  Range: " + string(maximum - minimum))
}

fn main() {
    let test_scores = [85, 92, 78, 96, 88, 91, 84, 89, 93, 87]
    calculate_statistics(test_scores)
}
```

## 🤖 AI/ML Examples

### Tensor Operations
```aqualua
// tensor_demo.aq
fn main() {
    print("🧠 Tensor Operations Demo")
    
    // Create tensors
    tensor weights = random([3, 4])
    tensor biases = zeros([4])
    tensor input_data = ones([3])
    
    print("Created tensors:")
    print("  Weights shape: [3, 4]")
    print("  Biases shape: [4]")
    print("  Input shape: [3]")
    
    // Matrix multiplication
    tensor output = matmul(transpose(weights), input_data)
    print("Matrix multiplication completed")
    
    // Add bias
    tensor final_output = add(output, biases)
    print("Added bias")
    
    // Apply activation
    tensor activated = relu(final_output)
    print("Applied ReLU activation")
    
    print("✅ Tensor operations completed successfully!")
}
```

### Simple Neural Network
```aqualua
// neural_network.aq
fn create_layer(input_size: int, output_size: int) -> layer {
    return dense(input_size, output_size, "relu")
}

fn forward_pass(input_data: tensor) -> tensor {
    // Create network layers
    layer layer1 = create_layer(784, 128)
    layer layer2 = create_layer(128, 64)
    layer output_layer = dense(64, 10, "softmax")
    
    // Forward propagation
    tensor hidden1 = layer1.forward(input_data)
    tensor hidden2 = layer2.forward(hidden1)
    tensor predictions = output_layer.forward(hidden2)
    
    return predictions
}

fn main() {
    print("🧠 Neural Network Demo")
    
    // Simulate input data (28x28 image flattened)
    tensor input_image = random([1, 784])
    
    // Run forward pass
    tensor predictions = forward_pass(input_image)
    
    print("✅ Neural network inference completed!")
    print("Output shape: [1, 10] (10 classes)")
}
```

### Machine Learning Pipeline
```aqualua
// ml_pipeline.aq
fn preprocess_data(raw_data: tensor) -> tensor {
    // Normalize data to [0, 1] range
    tensor normalized = div(raw_data, 255.0)
    return normalized
}

fn train_model(training_data: tensor, labels: tensor) {
    print("🏋️ Training model...")
    
    // Create model architecture
    layer conv1 = conv2d(32, [3, 3], "relu")
    layer conv2 = conv2d(64, [3, 3], "relu")
    layer flatten = flatten_layer()
    layer dense1 = dense(1024, 128, "relu")
    layer output = dense(128, 10, "softmax")
    
    // Training loop simulation
    for epoch in range(1, 11) {
        // Forward pass
        tensor features = conv1.forward(training_data)
        features = conv2.forward(features)
        features = flatten.forward(features)
        features = dense1.forward(features)
        tensor predictions = output.forward(features)
        
        // Calculate loss
        tensor loss = cross_entropy_loss(predictions, labels)
        
        print("Epoch " + string(epoch) + " - Loss: simulated")
    }
    
    print("✅ Model training completed!")
}

fn main() {
    print("🤖 Machine Learning Pipeline")
    
    // Simulate dataset (batch_size=32, height=28, width=28, channels=1)
    tensor raw_images = random([32, 28, 28, 1])
    tensor labels = random([32, 10])  // One-hot encoded labels
    
    // Preprocess data
    tensor processed_data = preprocess_data(raw_images)
    
    // Train model
    train_model(processed_data, labels)
}
```

## 🐍 Python Integration Examples

### Data Science with Pandas
```aqualua
// data_science.aq
fn main() {
    print("📊 Data Science with Python Integration")
    
    ast_exec("
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create sample dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 70000, 55000, 65000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing']
}

df = pd.DataFrame(data)
print('Dataset:')
print(df)

# Basic statistics
print('\\nStatistics:')
print(df.describe())

# Group by department
print('\\nAverage salary by department:')
dept_salary = df.groupby('department')['salary'].mean()
print(dept_salary)

# Create visualization
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df['department'].value_counts().plot(kind='bar')
plt.title('Employees by Department')

plt.subplot(1, 2, 2)
plt.scatter(df['age'], df['salary'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary')

plt.tight_layout()
plt.show()
")
    
    print("✅ Data analysis completed!")
}
```

### Machine Learning with Scikit-Learn
```aqualua
// sklearn_demo.aq
fn main() {
    print("🤖 Machine Learning with Scikit-Learn")
    
    ast_exec("
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Generate sample dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

print(f'Dataset shape: {X.shape}')
print(f'Number of classes: {len(np.unique(y))}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'\\nAccuracy: {accuracy:.3f}')

print('\\nClassification Report:')
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.feature_importances_
print(f'\\nTop 5 most important features:')
for i in np.argsort(feature_importance)[-5:]:
    print(f'  Feature {i}: {feature_importance[i]:.3f}')
")
    
    print("✅ Machine learning model trained and evaluated!")
}
```

## 🎮 Game Development Examples

### Simple Text Adventure
```aqualua
// text_adventure.aq
class Player {
    fn init(name: string) {
        this.name = name
        this.health = 100
        this.inventory = []
    }
    
    fn take_damage(damage: int) {
        this.health = this.health - damage
        if this.health < 0 {
            this.health = 0
        }
    }
    
    fn is_alive() -> bool {
        return this.health > 0
    }
}

fn show_status(player: Player) {
    print("=== Status ===")
    print("Name: " + player.name)
    print("Health: " + string(player.health))
    print("Inventory: " + string(len(player.inventory)) + " items")
}

fn main() {
    print("🎮 Welcome to AquaLua Adventure!")
    
    let player_name = input("Enter your name: ")
    let player = Player(player_name)
    
    print("Welcome, " + player.name + "!")
    
    let playing = true
    while playing && player.is_alive() {
        show_status(player)
        print("\nWhat do you want to do?")
        print("1. Explore")
        print("2. Rest")
        print("3. Quit")
        
        let choice = input("Enter choice (1-3): ")
        
        if choice == "1" {
            print("You explore the dungeon...")
            let encounter = randint(1, 3)
            if encounter == 1 {
                print("You found a treasure chest! +10 gold")
            } else if encounter == 2 {
                print("A monster attacks! You take 20 damage.")
                player.take_damage(20)
            } else {
                print("You find nothing interesting.")
            }
        } else if choice == "2" {
            print("You rest and recover 15 health.")
            player.health = player.health + 15
            if player.health > 100 {
                player.health = 100
            }
        } else if choice == "3" {
            playing = false
        }
    }
    
    if !player.is_alive() {
        print("💀 Game Over! You have died.")
    } else {
        print("👋 Thanks for playing!")
    }
}
```

### Pygame Integration
```aqualua
// pygame_demo.aq
fn main() {
    print("🎮 Pygame Integration Demo")
    
    ast_exec("
import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('AquaLua Pygame Demo')
clock = pygame.time.Clock()

# Game objects
player_x = SCREEN_WIDTH // 2
player_y = SCREEN_HEIGHT // 2
player_speed = 5

enemies = []
for _ in range(5):
    enemy_x = random.randint(0, SCREEN_WIDTH - 20)
    enemy_y = random.randint(0, SCREEN_HEIGHT - 20)
    enemies.append([enemy_x, enemy_y])

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Handle input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x < SCREEN_WIDTH - 20:
        player_x += player_speed
    if keys[pygame.K_UP] and player_y > 0:
        player_y -= player_speed
    if keys[pygame.K_DOWN] and player_y < SCREEN_HEIGHT - 20:
        player_y += player_speed
    
    # Move enemies
    for enemy in enemies:
        enemy[0] += random.randint(-2, 2)
        enemy[1] += random.randint(-2, 2)
        
        # Keep enemies on screen
        enemy[0] = max(0, min(SCREEN_WIDTH - 20, enemy[0]))
        enemy[1] = max(0, min(SCREEN_HEIGHT - 20, enemy[1]))
    
    # Draw everything
    screen.fill(WHITE)
    
    # Draw player
    pygame.draw.rect(screen, BLUE, (player_x, player_y, 20, 20))
    
    # Draw enemies
    for enemy in enemies:
        pygame.draw.rect(screen, RED, (enemy[0], enemy[1], 20, 20))
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print('Game closed!')
")
    
    print("✅ Pygame demo completed!")
}
```

## 📊 Data Processing Examples

### CSV Data Processing
```aqualua
// csv_processing.aq
fn main() {
    print("📊 CSV Data Processing")
    
    ast_exec("
import csv
import statistics

# Create sample CSV data
sample_data = [
    ['Name', 'Age', 'City', 'Salary'],
    ['Alice', '25', 'New York', '75000'],
    ['Bob', '30', 'San Francisco', '85000'],
    ['Charlie', '35', 'Chicago', '70000'],
    ['Diana', '28', 'Boston', '80000'],
    ['Eve', '32', 'Seattle', '90000']
]

# Write CSV file
with open('employees.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sample_data)

print('Created employees.csv')

# Read and process CSV
employees = []
with open('employees.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        employees.append({
            'name': row['Name'],
            'age': int(row['Age']),
            'city': row['City'],
            'salary': int(row['Salary'])
        })

print(f'Loaded {len(employees)} employees')

# Calculate statistics
ages = [emp['age'] for emp in employees]
salaries = [emp['salary'] for emp in employees]

print(f'Average age: {statistics.mean(ages):.1f}')
print(f'Average salary: ${statistics.mean(salaries):,.0f}')
print(f'Salary range: ${min(salaries):,} - ${max(salaries):,}')

# Group by city
cities = {}
for emp in employees:
    city = emp['city']
    if city not in cities:
        cities[city] = []
    cities[city].append(emp)

print('\\nEmployees by city:')
for city, city_employees in cities.items():
    avg_salary = statistics.mean([emp['salary'] for emp in city_employees])
    print(f'  {city}: {len(city_employees)} employees, avg salary ${avg_salary:,.0f}')
")
    
    print("✅ CSV processing completed!")
}
```

## 🌐 Web Development Examples

### Simple Web Server
```aqualua
// web_server.aq
fn main() {
    print("🌐 Simple Web Server")
    
    ast_exec("
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs

class AquaLuaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>AquaLua Web Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { color: #2c3e50; }
                    .info { background: #ecf0f1; padding: 20px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1 class='header'>🚀 AquaLua Web Server</h1>
                <div class='info'>
                    <p>This web server is powered by AquaLua!</p>
                    <p>Try these endpoints:</p>
                    <ul>
                        <li><a href='/api/status'>/api/status</a> - Server status</li>
                        <li><a href='/api/math?x=5&y=3'>/api/math?x=5&y=3</a> - Math operations</li>
                        <li><a href='/api/ai'>/api/ai</a> - AI demo</li>
                    </ul>
                </div>
            </body>
            </html>
            '''
            self.wfile.write(html.encode())
            
        elif parsed_path.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                'status': 'running',
                'language': 'AquaLua',
                'version': '1.0',
                'features': ['AI/ML', 'Web Server', 'Python Integration']
            }
            self.wfile.write(json.dumps(status, indent=2).encode())
            
        elif parsed_path.path == '/api/math':
            query_params = parse_qs(parsed_path.query)
            x = float(query_params.get('x', [0])[0])
            y = float(query_params.get('y', [0])[0])
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            result = {
                'input': {'x': x, 'y': y},
                'operations': {
                    'addition': x + y,
                    'subtraction': x - y,
                    'multiplication': x * y,
                    'division': x / y if y != 0 else 'undefined'
                }
            }
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        elif parsed_path.path == '/api/ai':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Simulate AI processing
            import random
            ai_result = {
                'model': 'AquaLua Neural Network',
                'prediction': random.choice(['cat', 'dog', 'bird']),
                'confidence': round(random.uniform(0.7, 0.99), 3),
                'processing_time': f'{random.randint(10, 100)}ms'
            }
            self.wfile.write(json.dumps(ai_result, indent=2).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'404 - Not Found')
    
    def log_message(self, format, *args):
        print(f'[{self.address_string()}] {format % args}')

# Start server
server_address = ('localhost', 8000)
httpd = HTTPServer(server_address, AquaLuaHandler)

print('🌐 AquaLua Web Server starting...')
print('📍 Server running at http://localhost:8000')
print('🛑 Press Ctrl+C to stop')

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print('\\n🛑 Server stopped')
    httpd.server_close()
")
}
```

## 📚 More Examples

### File I/O Operations
```aqualua
// file_operations.aq
fn main() {
    print("📁 File I/O Operations")
    
    ast_exec("
import os
import json

# Write text file
with open('sample.txt', 'w') as f:
    f.write('Hello from AquaLua!\\n')
    f.write('This is a sample text file.\\n')
    f.write('Created with Python integration.')

print('Created sample.txt')

# Read text file
with open('sample.txt', 'r') as f:
    content = f.read()
    print('File contents:')
    print(content)

# Write JSON file
data = {
    'language': 'AquaLua',
    'version': '1.0',
    'features': ['AI/ML', 'High Performance', 'Python Integration'],
    'author': 'AquaLua Team'
}

with open('config.json', 'w') as f:
    json.dump(data, f, indent=2)

print('Created config.json')

# Read JSON file
with open('config.json', 'r') as f:
    loaded_data = json.load(f)
    print('JSON data:')
    for key, value in loaded_data.items():
        print(f'  {key}: {value}')

# List files in directory
print('Files in current directory:')
for filename in os.listdir('.'):
    if os.path.isfile(filename):
        size = os.path.getsize(filename)
        print(f'  {filename} ({size} bytes)')
")
}
```

These examples demonstrate the power and versatility of AquaLua across different domains. Each example is self-contained and can be run independently to explore specific features of the language.