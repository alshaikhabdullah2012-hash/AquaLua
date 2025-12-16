"""
Aqualua Lexer - Tokenizes Aqualua source code
Part of the Python frontend for rapid development
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Iterator

class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers and keywords
    IDENTIFIER = auto()
    
    # Keywords
    FN = auto()
    LET = auto()
    CONST = auto()
    MODEL = auto()
    LAYER = auto()
    OPTIMIZER = auto()
    TRAIN = auto()
    STEP = auto()
    RETURN = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TYPE = auto()
    DEVICE = auto()
    TENSOR = auto()
    DATASET = auto()
    EPOCH = auto()
    OPTIMIZE = auto()
    LOG = auto()
    FOR = auto()
    IN = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRY = auto()
    EXCEPT = auto()
    FINALLY = auto()
    RAISE = auto()
    MATCH = auto()
    USING = auto()
    ON = auto()
    TRUE = auto()
    FALSE = auto()
    CLASS = auto()
    FUNCTION = auto()
    DEF = auto()
    NULL = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    MATMUL = auto()  # @
    
    # Comparison
    EQ = auto()      # ==
    NE = auto()      # !=
    LT = auto()      # <
    GT = auto()      # >
    LE = auto()      # <=
    GE = auto()      # >=
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()     # !
    
    # Assignment
    ASSIGN = auto()  # =
    
    # Delimiters
    LPAREN = auto()    # (
    RPAREN = auto()    # )
    LBRACE = auto()    # {
    RBRACE = auto()    # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()     # ,
    SEMICOLON = auto() # ;
    COLON = auto()     # :
    DOT = auto()       # .
    ARROW = auto()     # ->
    
    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    
    # Comments
    COMMENT = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class AqualuaLexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        
        # Keywords mapping
        self.keywords = {
            'fn': TokenType.FN,
            'let': TokenType.LET,
            'const': TokenType.CONST,
            'model': TokenType.MODEL,
            'layer': TokenType.LAYER,
            'optimizer': TokenType.OPTIMIZER,
            'train': TokenType.TRAIN,
            'step': TokenType.STEP,
            'return': TokenType.RETURN,
            'import': TokenType.IMPORT,
            'from': TokenType.FROM,
            'as': TokenType.AS,
            'type': TokenType.TYPE,
            # 'device': TokenType.DEVICE,  # Commented out - treat as identifier
            'tensor': TokenType.TENSOR,
            'dataset': TokenType.DATASET,
            'epoch': TokenType.EPOCH,
            'optimize': TokenType.OPTIMIZE,
            'log': TokenType.LOG,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'elif': TokenType.ELIF,
            'while': TokenType.WHILE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'try': TokenType.TRY,
            'except': TokenType.EXCEPT,
            'finally': TokenType.FINALLY,
            'raise': TokenType.RAISE,
            'match': TokenType.MATCH,
            'using': TokenType.USING,
            'on': TokenType.ON,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'class': TokenType.CLASS,
            'function': TokenType.FUNCTION,
            'def': TokenType.DEF,
            'null': TokenType.NULL,
        }
        
        # Token patterns - ORDER MATTERS! Longer patterns first
        self.patterns = [
            (r'#.*', TokenType.COMMENT),
            (r'\d+\.\d+', TokenType.FLOAT),
            (r'\d+', TokenType.INTEGER),
            (r'`[^`]*`', TokenType.STRING),
            (r'"(?:[^"\\]|\\.)*"', TokenType.STRING),
            (r"'(?:[^'\\]|\\.)*'", TokenType.STRING),
            (r'->', TokenType.ARROW),
            (r'==', TokenType.EQ),
            (r'!=', TokenType.NE),
            (r'<=', TokenType.LE),
            (r'>=', TokenType.GE),
            (r'<', TokenType.LT),
            (r'>', TokenType.GT),
            (r'@', TokenType.MATMUL),
            (r'\+', TokenType.PLUS),
            (r'-', TokenType.MINUS),
            (r'\*', TokenType.MULTIPLY),
            (r'/', TokenType.DIVIDE),
            (r'%', TokenType.MODULO),
            (r'=', TokenType.ASSIGN),
            (r'!', TokenType.NOT),
            (r'\(', TokenType.LPAREN),
            (r'\)', TokenType.RPAREN),
            (r'\{', TokenType.LBRACE),
            (r'\}', TokenType.RBRACE),
            (r'\[', TokenType.LBRACKET),
            (r'\]', TokenType.RBRACKET),
            (r',', TokenType.COMMA),
            (r';', TokenType.SEMICOLON),
            (r':', TokenType.COLON),
            (r'\.', TokenType.DOT),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
            (r'\n', TokenType.NEWLINE),
        ]
        
        # Compile patterns
        self.compiled_patterns = [(re.compile(pattern), token_type) 
                                 for pattern, token_type in self.patterns]
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self) -> None:
        if self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def skip_whitespace(self) -> None:
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def handle_string(self, quote_char: str) -> None:
        """Handle string literals with proper escape sequence support"""
        start_line = self.line
        start_column = self.column
        
        # Skip opening quote
        self.advance()
        
        value = quote_char
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                # Handle escape sequences
                value += self.current_char()
                self.advance()
                if self.current_char():
                    value += self.current_char()
                    self.advance()
            else:
                value += self.current_char()
                self.advance()
        
        if self.current_char() == quote_char:
            value += quote_char
            self.advance()
        
        # Create token
        token = Token(TokenType.STRING, value, start_line, start_column)
        self.tokens.append(token)
    
    def handle_backtick_string(self) -> None:
        """Handle multiline backtick strings for exec blocks"""
        start_line = self.line
        start_column = self.column
        
        # Skip opening backtick
        self.advance()
        
        value = '`'
        while self.current_char() and self.current_char() != '`':
            if self.current_char() == '\\':
                # Handle escape sequences in backtick strings too
                value += self.current_char()
                self.advance()
                if self.current_char():
                    value += self.current_char()
                    self.advance()
            else:
                value += self.current_char()
                self.advance()
        
        if self.current_char() == '`':
            value += '`'
            self.advance()
        
        # Create token
        token = Token(TokenType.STRING, value, start_line, start_column)
        self.tokens.append(token)
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if self.pos >= len(self.source):
                break
            
            # Handle strings specially to support escape sequences
            if self.current_char() == '`':
                self.handle_backtick_string()
                continue
            elif self.current_char() in ['"', "'"]:
                self.handle_string(self.current_char())
                continue
            
            # Try to match patterns
            matched = False
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(self.source, self.pos)
                if match:
                    value = match.group(0)
                    
                    # Skip comments
                    if token_type == TokenType.COMMENT:
                        self.pos = match.end()
                        continue
                    
                    # Handle keywords vs identifiers
                    if token_type == TokenType.IDENTIFIER:
                        token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                    
                    # Create token
                    token = Token(token_type, value, self.line, self.column)
                    self.tokens.append(token)
                    
                    # Update position
                    self.pos = match.end()
                    if value == '\n':
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += len(value)
                    
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Unexpected character '{self.current_char()}' at line {self.line}, column {self.column}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize Aqualua source code"""
    lexer = AqualuaLexer(source)
    return lexer.tokenize()

if __name__ == "__main__":
    # Test the lexer
    test_code = '''
    model MLP {
        l1: Linear(784, 256)
        l2: Linear(256, 10)
        
        fn forward(x: Tensor[B, 784]) -> Tensor[B, 10] {
            return l2(relu(l1(x)))
        }
    }
    
    let x = 42
    train model using Adam(lr=1e-3)
    '''
    
    tokens = tokenize(test_code)
    for token in tokens:
        print(f"{token.type.name:12} {token.value!r:15} Line {token.line:2} Col {token.column:2}")