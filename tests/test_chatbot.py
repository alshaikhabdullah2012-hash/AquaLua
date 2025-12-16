#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from aqualua_parser import parse
from aqualua_interpreter import AqualuaInterpreter

def test_chatbot():
    # Read the natural chatbot file
    with open('natural_chatbot.aq', 'r', encoding='utf-8') as f:
        source = f.read()

    print('Testing natural chatbot...')

    try:
        # Parse
        ast = parse(source)
        print('Parsing successful')
        
        # Interpret
        interpreter = AqualuaInterpreter()
        
        # Mock input function for testing - simulate user saying "hello" then "bye"
        inputs = ["hello", "bye"]
        input_count = 0
        
        def mock_input(prompt=''):
            nonlocal input_count
            if input_count < len(inputs):
                user_input = inputs[input_count]
                print(f"{prompt}{user_input}")
                input_count += 1
                return user_input
            else:
                return "bye"  # Default to bye to end conversation
        
        interpreter.environment.define('input', mock_input)
        
        print('Starting interpretation...')
        interpreter.interpret(ast)
        print('Interpretation complete')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_chatbot()