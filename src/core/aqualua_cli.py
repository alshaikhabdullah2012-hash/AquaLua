#!/usr/bin/env python3
"""
Aqualua CLI - Command Line Interface for running Aqualua files
"""

import sys
import os

def run_aqualua_file(filename):
    """Run an Aqualua file"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from aqualua_parser import parse
        from aqualua_interpreter import AqualuaInterpreter
        
        # Read and parse the file
        with open(filename, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        print(f"Running Aqualua program: {filename}")
        print("=" * 50)
        
        # Parse the source code
        ast = parse(source_code)
        
        # Create interpreter and run
        interpreter = AqualuaInterpreter()
        interpreter.interpret(ast)
        
        print("\n" + "=" * 50)
        print("Program completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python aqualua_cli.py <filename.aq>")
        return 1
    
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return 1
    
    return run_aqualua_file(filename)

if __name__ == "__main__":
    exit_code = main()
    # Only wait for input if running directly, not from IDE
    if len(sys.argv) >= 2 and not os.environ.get('AQUALUA_IDE_RUN'):
        input("\nPress Enter to close...")  # Keep terminal open
    sys.exit(exit_code)