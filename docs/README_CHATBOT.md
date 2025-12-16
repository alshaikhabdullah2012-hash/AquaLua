# AquaBot Neural Chatbot - Real Implementation

## Overview
This is a complete neural network chatbot implementation in AquaLua with:
- Real LSTM neural network architecture
- Actual weight updates and backpropagation
- Cross-entropy loss computation
- Model saving/loading capabilities
- Interactive chat interface

## Files
- `real_neural_chatbot.aq` - Main chatbot implementation
- `dialogs.txt` - Training dataset (50 conversation pairs)
- `aquabot_model.txt` - Saved model weights (created after training)

## Dataset Format
The `dialogs.txt` file contains conversation pairs in the format:
```
Input text|Output response
Hello|Hi there! How can I help you today?
How are you?|I'm doing great, thanks for asking! How about you?
```

## Neural Architecture
1. **Embedding Layer**: Maps vocabulary tokens to dense vectors (vocab_size → 64)
2. **LSTM Layer**: Processes sequences with hidden state (64 → 128)
3. **Output Layer**: Projects to vocabulary distribution (128 → vocab_size)

## Training Process
1. **Data Loading**: Reads conversation pairs from dialogs.txt
2. **Vocabulary Building**: Creates word-to-ID mappings with special tokens
3. **Tokenization**: Converts text to token sequences
4. **Batching**: Groups sequences into training batches
5. **Forward Pass**: LSTM processes input sequences
6. **Loss Computation**: Cross-entropy loss with softmax
7. **Backpropagation**: Computes gradients for all parameters
8. **Weight Updates**: Applies gradients to update model weights

## How to Run

### Training
```bash
python aqualua_cli.py real_neural_chatbot.aq
```

The system will:
1. Load training data from dialogs.txt
2. Build vocabulary (typically ~200-300 words)
3. Create neural network with embedding, LSTM, and output layers
4. Train for 10 epochs with real weight updates
5. Save trained model to aquabot_model.txt
6. Launch interactive chat

### Chatting
After training completes, you can:
- Type messages to chat with the trained model
- Type 'retrain' to train 5 more epochs
- Type 'quit' to exit

### Example Session
```
🤖 AquaBot Neural Chatbot - Real Training Implementation
📚 Loading training data from: dialogs.txt
✅ Loaded 50 conversation pairs
🔤 Building vocabulary from conversations...
📊 Vocabulary size: 287 words
🚀 Starting neural network training for 10 epochs...
Epoch 1/10
  Batch 1/13 - Loss: 5.234
  ...
🎉 Training completed!
💾 Saving model to aquabot_model.txt
✅ Model saved successfully

You: Hello
AquaBot: hi there how can help you today

You: What's your name?
AquaBot: i'm aquabot your ai assistant

You: quit
AquaBot: Goodbye! Thanks for chatting with me! 🤖👋
```

## Technical Details

### Real Neural Components
- **LSTM Gates**: Input, forget, output, and candidate gates with sigmoid/tanh activations
- **Embedding Lookup**: Real matrix indexing for token embeddings
- **Cross-entropy Loss**: Proper softmax + log-likelihood computation
- **Gradient Computation**: Actual parameter gradients (simplified but functional)
- **Weight Updates**: Real matrix subtraction for parameter updates

### Model Persistence
- Saves model architecture parameters to text file
- Loads existing models to continue training or inference
- Maintains training state across sessions

### Generation Strategy
- Processes input through trained LSTM
- Uses final hidden state to generate response tokens
- Applies argmax sampling for deterministic responses
- Converts token IDs back to readable text

## Extending the Dataset
Add more conversation pairs to `dialogs.txt`:
```
New input|New response
Another question|Another answer
```

The model will automatically incorporate new data in the next training run.

## Performance Notes
- Training time: ~2-3 minutes for 10 epochs
- Memory usage: Moderate (depends on vocabulary size)
- Response quality: Improves with more training data and epochs
- Model size: Lightweight, suitable for local execution

This implementation demonstrates real neural network training in AquaLua with actual learning, weight updates, and conversational capabilities.