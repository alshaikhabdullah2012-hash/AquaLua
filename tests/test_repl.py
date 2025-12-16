#!/usr/bin/env python3
"""
Test REPL functionality directly
"""

import sys
sys.path.append('.')

from aqualua_cli import AqualuaREPL

# Create REPL instance
repl = AqualuaREPL()

# Test commands
test_commands = [
    "let x = 4",
    "print(x)",
    'print("hello")',
    "let y = 5",
    "x + y"
]

print("Testing Aqualua REPL:")
print("=" * 30)

for cmd in test_commands:
    print(f"\naqualua> {cmd}")
    try:
        repl.execute_line(cmd)
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 30)
print("REPL test complete!")