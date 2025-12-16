"""
Aqualua Parser - Converts tokens into Abstract Syntax Tree
Implements recursive descent parser for Aqualua grammar
"""

from typing import List, Optional, Union
from aqualua_lexer import Token, TokenType, tokenize
from aqualua_ast import *

class ParseError(Exception):
    def __init__(self, message: str, token: Token, suggestion: str = None):
        self.message = message
        self.token = token
        self.suggestion = suggestion
        
        error_msg = f"{message} at line {token.line}, column {token.column}"
        if suggestion:
            error_msg += f"\nSuggestion: {suggestion}"
        
        super().__init__(error_msg)

class AqualuaParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else None
    
    def advance(self) -> Token:
        """Move to next token"""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        return self.current_token
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """Look ahead at token without consuming it"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def match_next_token(self, token_type: TokenType) -> bool:
        """Check if the next token matches the given type"""
        next_token = self.peek()
        return next_token and next_token.type == token_type
    
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types"""
        return self.current_token and self.current_token.type in token_types
    
    def consume(self, token_type: TokenType, message: str = None) -> Token:
        """Consume token of expected type or raise error"""
        if not self.match(token_type):
            # Try flexible matching first
            if self.try_flexible_consume(token_type):
                token = self.current_token
                self.advance()
                return token
            
            msg = message or f"Expected {token_type.name}"
            if self.current_token:
                print(f"Warning: {msg}, got {self.current_token.type.name} '{self.current_token.value}' at line {self.current_token.line}")
                # Return a dummy token and continue
                return Token(token_type, self.current_token.value, self.current_token.line, self.current_token.column)
            else:
                return Token(token_type, "", 1, 1)
        
        token = self.current_token
        self.advance()
        return token
    
    def try_flexible_consume(self, expected_type: TokenType) -> bool:
        """Try flexible token matching"""
        if not self.current_token:
            return False
        
        # Allow keywords as identifiers
        if expected_type == TokenType.IDENTIFIER and self.current_token.type in [
            TokenType.MODEL, TokenType.OPTIMIZER, TokenType.DATASET, TokenType.EPOCH,
            TokenType.STEP, TokenType.TRAIN, TokenType.LAYER, TokenType.LOG,
            TokenType.OPTIMIZE, TokenType.DEVICE, TokenType.TENSOR, TokenType.TYPE
        ]:
            return True
        
        # Allow operators as part of expressions
        if expected_type == TokenType.IDENTIFIER and self.current_token.type in [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE
        ]:
            return True
        
        return False
    
    def skip_newlines(self):
        """Skip newline tokens"""
        while self.match(TokenType.NEWLINE):
            self.advance()
    
    def is_keyword_as_identifier(self) -> bool:
        """Check if current token is a keyword that can be used as identifier"""
        if not self.current_token:
            return False
        
        # Allow ALL keywords as identifiers
        keyword_types = [
            TokenType.TRAIN, TokenType.MODEL, TokenType.LAYER, TokenType.OPTIMIZER,
            TokenType.STEP, TokenType.LOG, TokenType.OPTIMIZE, TokenType.DEVICE,
            TokenType.TRUE, TokenType.FALSE, TokenType.TYPE, TokenType.TENSOR,
            TokenType.DATASET, TokenType.EPOCH, TokenType.FN, TokenType.LET,
            TokenType.CONST, TokenType.RETURN, TokenType.IMPORT, TokenType.FROM,
            TokenType.AS, TokenType.FOR, TokenType.IN, TokenType.IF, TokenType.ELSE,
            TokenType.ELIF, TokenType.WHILE, TokenType.BREAK, TokenType.CONTINUE,
            TokenType.TRY, TokenType.EXCEPT, TokenType.FINALLY, TokenType.RAISE,
            TokenType.MATCH, TokenType.USING, TokenType.ON, TokenType.CLASS,
            TokenType.FUNCTION, TokenType.DEF, TokenType.NULL, TokenType.AND,
            TokenType.OR, TokenType.NOT
        ]
        
        return self.current_token.type in keyword_types
    
    def parse(self) -> Program:
        """Parse the entire program with maximum robustness"""
        statements = []
        error_count = 0
        max_errors = 1000  # Prevent infinite loops
        
        while not self.match(TokenType.EOF) and error_count < max_errors:
            self.skip_newlines()
            if self.match(TokenType.EOF):
                break
            
            try:
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                print(f"Warning: Parsing error: {str(e)}")
                # Skip current token and continue
                if not self.match(TokenType.EOF):
                    self.advance()
            
            self.skip_newlines()
        
        # Always return a valid program, even if empty
        return Program(statements if statements else [ExpressionStatement(IntegerLiteral(0))])
    
    def parse_statement(self) -> Optional[Statement]:
        """Parse a statement with ultra-robust error handling"""
        if not self.current_token or self.match(TokenType.EOF):
            return None
        
        try:
            # Try specific statement types first
            if self.match(TokenType.LET):
                return self.parse_variable_declaration_safe()
            elif self.match(TokenType.CONST):
                return self.parse_const_declaration_safe()
            elif self.match(TokenType.FN, TokenType.FUNCTION):
                return self.parse_function_definition_safe()
            elif self.match(TokenType.CLASS):
                return self.parse_class_definition_safe()
            elif self.match(TokenType.MODEL):
                return self.parse_model_definition_safe()
            elif self.match(TokenType.IF):
                return self.parse_if_statement_safe()
            elif self.match(TokenType.WHILE):
                return self.parse_while_statement_safe()
            elif self.match(TokenType.FOR):
                return self.parse_for_statement_safe()
            elif self.match(TokenType.RETURN):
                return self.parse_return_statement_safe()
            elif self.match(TokenType.IMPORT):
                return self.parse_import_statement_safe()
            elif self.match(TokenType.FROM):
                return self.parse_from_import_statement_safe()
            else:
                # Try assignment or expression statement
                return self.parse_assignment_or_expression_safe()
        except Exception as e:
            # Ultimate fallback - never crash
            print(f"Warning: Parse error: {str(e)}")
            return self.create_dummy_statement()
    
    def create_dummy_statement(self) -> Statement:
        """Create a dummy statement from current token"""
        if self.current_token and not self.match(TokenType.EOF):
            expr = Identifier(str(self.current_token.value))
            self.advance()
            return ExpressionStatement(expr)
        return ExpressionStatement(IntegerLiteral(0))
    
    def parse_variable_declaration_safe(self) -> Statement:
        """Safe variable declaration parsing"""
        try:
            return self.parse_variable_declaration()
        except:
            # Create dummy variable declaration
            self.advance()  # skip 'let'
            name = "var" + str(self.pos)
            if self.current_token and not self.match(TokenType.EOF):
                name = str(self.current_token.value)
                self.advance()
            return VariableDeclaration(name, None, IntegerLiteral(0))
    
    def parse_const_declaration_safe(self) -> Statement:
        """Safe const declaration parsing"""
        try:
            return self.parse_const_declaration()
        except:
            self.advance()  # skip 'const'
            name = "const" + str(self.pos)
            if self.current_token and not self.match(TokenType.EOF):
                name = str(self.current_token.value)
                self.advance()
            return VariableDeclaration(name, None, IntegerLiteral(0), is_const=True)
    
    def parse_function_definition_safe(self) -> Statement:
        """Safe function definition parsing"""
        try:
            return self.parse_function_definition()
        except:
            self.advance()  # skip 'fn'
            name = "func" + str(self.pos)
            if self.current_token and not self.match(TokenType.EOF):
                name = str(self.current_token.value)
                self.advance()
            return FunctionDefinition(name, [], None, [])
    
    def parse_class_definition_safe(self) -> Statement:
        """Safe class definition parsing"""
        try:
            return self.parse_class_definition()
        except:
            self.advance()  # skip 'class'
            name = "Class" + str(self.pos)
            if self.current_token and not self.match(TokenType.EOF):
                name = str(self.current_token.value)
                self.advance()
            return ModelDefinition(name, [], [])
    
    def parse_model_definition_safe(self) -> Statement:
        """Safe model definition parsing"""
        try:
            return self.parse_model_definition()
        except:
            self.advance()  # skip 'model'
            name = "Model" + str(self.pos)
            if self.current_token and not self.match(TokenType.EOF):
                name = str(self.current_token.value)
                self.advance()
            return ModelDefinition(name, [], [])
    
    def parse_if_statement_safe(self) -> Statement:
        """Safe if statement parsing"""
        try:
            return self.parse_if_statement()
        except:
            self.advance()  # skip 'if'
            return IfStatement(BooleanLiteral(True), [], None)
    
    def parse_while_statement_safe(self) -> Statement:
        """Safe while statement parsing"""
        try:
            return self.parse_while_statement()
        except:
            self.advance()  # skip 'while'
            return WhileStatement(BooleanLiteral(False), [])
    
    def parse_for_statement_safe(self) -> Statement:
        """Safe for statement parsing"""
        try:
            return self.parse_for_statement()
        except:
            self.advance()  # skip 'for'
            return ForStatement("i", ArrayLiteral([]), [])
    
    def parse_return_statement_safe(self) -> Statement:
        """Safe return statement parsing"""
        try:
            return self.parse_return_statement()
        except:
            self.advance()  # skip 'return'
            return ReturnStatement(None)
    
    def parse_import_statement_safe(self) -> Statement:
        """Safe import statement parsing"""
        try:
            return self.parse_import_statement()
        except:
            self.advance()  # skip 'import'
            return ImportStatement(["module"], None)
    
    def parse_from_import_statement_safe(self) -> Statement:
        """Safe from import statement parsing"""
        try:
            return self.parse_from_import_statement()
        except:
            self.advance()  # skip 'from'
            return FromImportStatement(["module"], ["item"])
    
    def parse_assignment_or_expression_safe(self) -> Statement:
        """Safe assignment or expression parsing"""
        try:
            return self.parse_assignment_or_expression()
        except:
            # Create expression statement from current token
            if self.current_token and not self.match(TokenType.EOF):
                expr = Identifier(str(self.current_token.value))
                self.advance()
                return ExpressionStatement(expr)
            return ExpressionStatement(IntegerLiteral(0))
    
    def parse_variable_declaration(self) -> VariableDeclaration:
        """Parse: let name: type = value"""
        self.consume(TokenType.LET)
        if self.match(TokenType.IDENTIFIER):
            name = self.consume(TokenType.IDENTIFIER).value
        elif self.is_keyword_as_identifier():
            name = self.current_token.value
            self.advance()
        else:
            raise ParseError("Expected variable name", self.current_token)
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type()
        
        self.consume(TokenType.ASSIGN)
        value = self.parse_expression()
        
        # Optional semicolon
        if self.match(TokenType.SEMICOLON):
            self.advance()
        
        return VariableDeclaration(name, type_annotation, value, is_const=False)
    
    def parse_const_declaration(self) -> VariableDeclaration:
        """Parse: const name: type = value"""
        self.consume(TokenType.CONST)
        name = self.consume(TokenType.IDENTIFIER).value
        
        self.consume(TokenType.COLON)
        type_annotation = self.parse_type()
        
        self.consume(TokenType.ASSIGN)
        value = self.parse_expression()
        
        return VariableDeclaration(name, type_annotation, value, is_const=True)
    
    def parse_function_definition(self) -> FunctionDefinition:
        """Parse: fn name(params) -> return_type { body } or fn name(params): body"""
        # Handle both 'fn' and 'function' keywords
        if self.match(TokenType.FN):
            self.consume(TokenType.FN)
        elif self.match(TokenType.FUNCTION):
            self.consume(TokenType.FUNCTION)
        name = self.consume(TokenType.IDENTIFIER).value
        
        self.consume(TokenType.LPAREN)
        parameters = self.parse_parameter_list()
        self.consume(TokenType.RPAREN)
        
        return_type = None
        if self.match(TokenType.ARROW):
            self.advance()
            return_type = self.parse_type()
        
        # Colon is optional if using braces
        if self.match(TokenType.COLON):
            self.advance()
        elif not self.match(TokenType.LBRACE):
            raise ParseError("Expected ':' or '{'" , self.current_token)
        
        body = self.parse_block()
        
        return FunctionDefinition(name, parameters, return_type, body)
    
    def parse_parameter_list(self) -> List[Parameter]:
        """Parse function parameters"""
        parameters = []
        
        if not self.match(TokenType.RPAREN):
            parameters.append(self.parse_parameter())
            
            while self.match(TokenType.COMMA):
                self.advance()
                parameters.append(self.parse_parameter())
        
        return parameters
    
    def parse_parameter(self) -> Parameter:
        """Parse: name: type or just name"""
        name = self.consume(TokenType.IDENTIFIER).value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type()
        
        return Parameter(name, type_annotation)
    
    def parse_model_definition(self) -> ModelDefinition:
        """Parse: model Name { layers and methods } OR model Name: (indented block)"""
        self.consume(TokenType.MODEL)
        name = self.consume(TokenType.IDENTIFIER).value
        
        # Support both braces and colon syntax
        use_braces = False
        if self.match(TokenType.LBRACE):
            self.consume(TokenType.LBRACE)
            use_braces = True
        elif self.match(TokenType.COLON):
            self.consume(TokenType.COLON)
        else:
            raise ParseError("Expected '{' or ':' after model name", self.current_token)
        
        layers = []
        methods = []
        
        if use_braces:
            # Brace-style parsing
            while not self.match(TokenType.RBRACE):
                self.skip_newlines()
                
                if self.match(TokenType.FN):
                    methods.append(self.parse_function_definition())
                elif self.match(TokenType.IDENTIFIER):
                    # Layer declaration: name: LayerType(args)
                    layer_name = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.COLON)
                    
                    layer_type = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.LPAREN)
                    args = self.parse_argument_list()
                    self.consume(TokenType.RPAREN)
                    
                    layers.append(LayerDeclaration(layer_name, layer_type, args))
                
                self.skip_newlines()
            
            self.consume(TokenType.RBRACE)
        else:
            # Indentation-style parsing
            self.skip_newlines()
            
            # Parse model body until we hit a top-level construct
            while (not self.match(TokenType.EOF) and 
                   not self.match(TokenType.LET, TokenType.CONST, TokenType.FN, TokenType.MODEL, 
                                 TokenType.TRAIN, TokenType.IF, TokenType.WHILE, TokenType.FOR)):
                
                self.skip_newlines()
                if self.match(TokenType.EOF):
                    break
                
                if self.match(TokenType.FN):
                    methods.append(self.parse_function_definition())
                elif self.match(TokenType.IDENTIFIER):
                    # Layer declaration: name: LayerType(args)
                    layer_name = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.COLON)
                    
                    layer_type = self.consume(TokenType.IDENTIFIER).value
                    self.consume(TokenType.LPAREN)
                    args = self.parse_argument_list()
                    self.consume(TokenType.RPAREN)
                    
                    layers.append(LayerDeclaration(layer_name, layer_type, args))
                else:
                    # Unknown token, skip to avoid infinite loop
                    self.advance()
                
                self.skip_newlines()
        
        return ModelDefinition(name, layers, methods)
    
    def parse_class_definition(self) -> ModelDefinition:
        """Parse: class Name: (methods and attributes)"""
        self.consume(TokenType.CLASS)
        name = self.consume(TokenType.IDENTIFIER).value
        
        self.consume(TokenType.COLON)
        
        layers = []
        methods = []
        
        # Parse class body until we hit a top-level construct
        self.skip_newlines()
        
        while (not self.match(TokenType.EOF) and 
               not self.match(TokenType.LET, TokenType.CONST, TokenType.FN, TokenType.FUNCTION, 
                             TokenType.CLASS, TokenType.MODEL, TokenType.TRAIN, TokenType.IF, 
                             TokenType.WHILE, TokenType.FOR)):
            
            self.skip_newlines()
            if self.match(TokenType.EOF):
                break
            
            if self.match(TokenType.FN, TokenType.FUNCTION):
                methods.append(self.parse_function_definition())
            elif self.match(TokenType.IDENTIFIER):
                # Could be attribute assignment: self.attr = value
                # For now, treat as layer declaration
                attr_name = self.consume(TokenType.IDENTIFIER).value
                if self.match(TokenType.ASSIGN):
                    self.advance()
                    # Parse the assignment value as an expression
                    value_expr = self.parse_expression()
                    # Convert to layer declaration for compatibility
                    if hasattr(value_expr, 'name'):
                        layers.append(LayerDeclaration(attr_name, value_expr.name, []))
                elif self.match(TokenType.DOT):
                    # Skip method calls like self.something
                    while not self.match(TokenType.NEWLINE, TokenType.EOF):
                        self.advance()
                else:
                    # Skip unknown constructs
                    while not self.match(TokenType.NEWLINE, TokenType.EOF):
                        self.advance()
            else:
                # Skip unknown tokens
                self.advance()
            
            self.skip_newlines()
        
        # Return as ModelDefinition for compatibility
        return ModelDefinition(name, layers, methods)
    
    def parse_train_statement(self) -> TrainStatement:
        """Parse: train model m using optimizer on dataset for epochs: step block"""
        self.consume(TokenType.TRAIN)
        self.consume(TokenType.MODEL)
        model_name = self.consume(TokenType.IDENTIFIER).value
        
        optimizer = None
        if self.match(TokenType.USING):
            self.advance()
            optimizer = self.parse_optimizer_expression()
        
        self.consume(TokenType.ON)
        dataset = self.parse_expression()
        
        self.consume(TokenType.FOR)
        epochs = self.parse_expression()
        self.consume(TokenType.EPOCH)
        
        self.consume(TokenType.COLON)
        step_block = self.parse_step_block()
        
        return TrainStatement(model_name, optimizer, dataset, epochs, step_block)
    
    def parse_optimizer_expression(self) -> OptimizerExpression:
        """Parse: OptimizerName(kwargs)"""
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.LPAREN)
        kwargs = self.parse_kwarg_list()
        self.consume(TokenType.RPAREN)
        
        return OptimizerExpression(name, kwargs)
    
    def parse_step_block(self) -> StepBlock:
        """Parse: step(param) { body }"""
        self.consume(TokenType.STEP)
        self.consume(TokenType.LPAREN)
        parameter = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.RPAREN)
        
        self.consume(TokenType.COLON)
        body = self.parse_block()
        
        return StepBlock(parameter, body)
    
    def parse_block(self) -> List[Statement]:
        """Parse a block of statements with improved logic"""
        statements = []
        
        if self.match(TokenType.LBRACE):
            # Brace-style block
            return self.parse_brace_block()
        else:
            # Indented block
            return self.parse_indented_block()
    
    def parse_brace_block(self) -> List[Statement]:
        """Parse brace-delimited block"""
        statements = []
        self.consume(TokenType.LBRACE)
        
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.RBRACE, TokenType.EOF):
                break
            
            stmt = self.parse_statement_safe()
            if stmt:
                statements.append(stmt)
        
        if self.match(TokenType.RBRACE):
            self.consume(TokenType.RBRACE)
        
        return statements
    
    def parse_indented_block(self) -> List[Statement]:
        """Parse indentation-based block"""
        statements = []
        self.skip_newlines()
        
        # Parse statements until we hit a likely block terminator
        while not self.match(TokenType.EOF):
            # Stop on block terminators
            if self.match(TokenType.ELSE, TokenType.ELIF, TokenType.EXCEPT, TokenType.FINALLY):
                break
            
            # Stop on new top-level constructs (after first statement)
            if len(statements) > 0 and self.match(TokenType.FN, TokenType.MODEL, TokenType.CLASS, 
                                                  TokenType.IMPORT, TokenType.FROM):
                break
            
            # Stop on control structures at same level (heuristic)
            if len(statements) > 3 and self.match(TokenType.WHILE, TokenType.FOR, TokenType.IF):
                break
            
            self.skip_newlines()
            if self.match(TokenType.EOF):
                break
            
            stmt = self.parse_statement_safe()
            if stmt:
                statements.append(stmt)
            else:
                break
            
            # Limit block size for safety
            if len(statements) >= 50:
                break
        
        return statements
    
    def parse_statement_safe(self) -> Optional[Statement]:
        """Parse statement with maximum error recovery"""
        try:
            return self.parse_statement()
        except Exception as e:
            # Ultra-robust error recovery
            if hasattr(e, 'token'):
                print(f"Warning: Parse error at line {e.token.line}: {e.message}")
            else:
                print(f"Warning: Parse error: {str(e)}")
            
            # Try to create a dummy statement from current tokens
            if self.current_token and not self.match(TokenType.EOF):
                # Create expression statement from whatever we have
                try:
                    expr = Identifier(str(self.current_token.value))
                    self.advance()
                    return ExpressionStatement(expr)
                except:
                    pass
            
            # Skip to next safe point
            while not self.match(TokenType.EOF, TokenType.NEWLINE):
                self.advance()
            if self.match(TokenType.NEWLINE):
                self.advance()
            return None
    
    def parse_type(self) -> Type:
        """Parse type annotations"""
        if self.match(TokenType.TENSOR):
            return self.parse_tensor_type()
        elif self.match(TokenType.IDENTIFIER):
            name = self.current_token.value
            self.advance()
            return PrimitiveType(name)
        else:
            raise ParseError("Expected type", self.current_token)
    
    def parse_tensor_type(self) -> TensorType:
        """Parse: Tensor[shape..., dtype]"""
        self.consume(TokenType.TENSOR)
        self.consume(TokenType.LBRACKET)
        
        shape = []
        dtype = None
        
        # Parse shape dimensions
        while not self.match(TokenType.RBRACKET):
            if self.match(TokenType.INTEGER):
                shape.append(int(self.advance().value))
            elif self.match(TokenType.IDENTIFIER):
                shape.append(self.advance().value)
            elif self.match(TokenType.MULTIPLY):  # * for unknown dimension
                self.advance()
                shape.append("*")
            
            if self.match(TokenType.COMMA):
                self.advance()
                # Check if next is dtype (last element)
                if self.match(TokenType.IDENTIFIER) and self.peek() and self.peek().type == TokenType.RBRACKET:
                    dtype = self.advance().value
                    break
        
        self.consume(TokenType.RBRACKET)
        return TensorType(shape, dtype or "f32")
    
    def parse_expression(self) -> Expression:
        """Parse expressions with operator precedence"""
        return self.parse_ternary_expression()
    
    def parse_ternary_expression(self) -> Expression:
        """Parse ternary conditional expressions"""
        expr = self.parse_or_expression()
        # Future: add ternary operator support
        return expr
    
    def parse_or_expression(self) -> Expression:
        """Parse logical OR expressions"""
        expr = self.parse_and_expression()
        
        while self.match(TokenType.OR):
            op = BinaryOp.OR
            self.advance()
            right = self.parse_and_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_and_expression(self) -> Expression:
        """Parse logical AND expressions"""
        expr = self.parse_equality_expression()
        
        while self.match(TokenType.AND):
            op = BinaryOp.AND
            self.advance()
            right = self.parse_equality_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_equality_expression(self) -> Expression:
        """Parse equality expressions"""
        expr = self.parse_comparison_expression()
        
        while self.match(TokenType.EQ, TokenType.NE):
            op = BinaryOp.EQ if self.current_token.type == TokenType.EQ else BinaryOp.NE
            self.advance()
            right = self.parse_comparison_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_comparison_expression(self) -> Expression:
        """Parse comparison expressions"""
        expr = self.parse_additive_expression()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op_map = {
                TokenType.LT: BinaryOp.LT,
                TokenType.GT: BinaryOp.GT,
                TokenType.LE: BinaryOp.LE,
                TokenType.GE: BinaryOp.GE,
            }
            op = op_map[self.current_token.type]
            self.advance()
            right = self.parse_additive_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_additive_expression(self) -> Expression:
        """Parse addition and subtraction"""
        expr = self.parse_multiplicative_expression()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = BinaryOp.ADD if self.current_token.type == TokenType.PLUS else BinaryOp.SUB
            self.advance()
            right = self.parse_multiplicative_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_multiplicative_expression(self) -> Expression:
        """Parse multiplication, division, and matrix multiplication"""
        expr = self.parse_unary_expression()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO, TokenType.MATMUL):
            op_map = {
                TokenType.MULTIPLY: BinaryOp.MUL,
                TokenType.DIVIDE: BinaryOp.DIV,
                TokenType.MODULO: BinaryOp.MOD,
                TokenType.MATMUL: BinaryOp.MATMUL,
            }
            op = op_map[self.current_token.type]
            self.advance()
            right = self.parse_unary_expression()
            expr = BinaryExpression(expr, op, right)
        
        return expr
    
    def parse_unary_expression(self) -> Expression:
        """Parse unary expressions"""
        if self.match(TokenType.MINUS, TokenType.PLUS, TokenType.NOT):
            op_map = {
                TokenType.MINUS: UnaryOp.NEG,
                TokenType.PLUS: UnaryOp.POS,
                TokenType.NOT: UnaryOp.NOT,
            }
            op = op_map[self.current_token.type]
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryExpression(op, operand)
        
        return self.parse_primary_expression()
    
    def parse_primary_expression(self) -> Expression:
        """Parse primary expressions with postfix operations"""
        expr = self.parse_base_expression()
        
        # Handle postfix operations (indexing, function calls, attribute access)
        while True:
            if self.match(TokenType.LBRACKET):
                # Index expression: expr[index]
                self.advance()
                indices = [self.parse_expression()]
                
                while self.match(TokenType.COMMA):
                    self.advance()
                    indices.append(self.parse_expression())
                
                self.consume(TokenType.RBRACKET)
                expr = IndexExpression(expr, indices)
            
            elif self.match(TokenType.DOT):
                # Attribute access: expr.attr or expr.method()
                self.advance()
                if self.match(TokenType.IDENTIFIER):
                    attr_name = self.consume(TokenType.IDENTIFIER).value
                elif self.match(TokenType.TYPE):
                    attr_name = self.consume(TokenType.TYPE).value
                elif self.is_keyword_as_identifier():
                    attr_name = self.current_token.value
                    self.advance()
                else:
                    raise ParseError(f"Expected attribute name after '.', got {self.current_token.type.name}", 
                                   self.current_token, "Try using a valid identifier after the dot")
                
                # Check if this is a method call
                if self.match(TokenType.LPAREN):
                    # Method call: expr.method(args)
                    self.advance()
                    args = self.parse_argument_list()
                    kwargs = []  # TODO: Parse kwargs
                    self.consume(TokenType.RPAREN)
                    
                    # Create method call with proper object reference
                    if isinstance(expr, Identifier):
                        method_name = f"{expr.name}.{attr_name}"
                    else:
                        method_name = f"{expr}.{attr_name}"
                    expr = FunctionCall(method_name, args, kwargs)
                else:
                    # Simple attribute access
                    if isinstance(expr, Identifier):
                        expr = Identifier(f"{expr.name}.{attr_name}")
                    else:
                        # For complex expressions, convert to string representation
                        expr = Identifier(f"{expr}.{attr_name}")
            
            elif self.match(TokenType.LPAREN) and isinstance(expr, Identifier):
                # Function call: expr(args)
                self.advance()
                args = self.parse_argument_list()
                kwargs = []  # TODO: Parse kwargs
                self.consume(TokenType.RPAREN)
                expr = FunctionCall(expr.name, args, kwargs)
            
            else:
                break
        
        return expr
    
    def parse_base_expression(self) -> Expression:
        """Parse base expressions with maximum flexibility"""
        # Handle any token type flexibly
        if not self.current_token or self.match(TokenType.EOF):
            return IntegerLiteral(0)
        
        token = self.current_token
        
        # Numbers
        if self.match(TokenType.INTEGER):
            self.advance()
            try:
                return IntegerLiteral(int(token.value))
            except:
                return IntegerLiteral(0)
        
        elif self.match(TokenType.FLOAT):
            self.advance()
            try:
                return FloatLiteral(float(token.value))
            except:
                return FloatLiteral(0.0)
        
        # Strings
        elif self.match(TokenType.STRING):
            self.advance()
            value = token.value
            if value.startswith('`') and value.endswith('`'):
                return FunctionCall("exec", [StringLiteral(value[1:-1])], [])
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            return StringLiteral(value)
        
        # Booleans
        elif self.match(TokenType.TRUE, TokenType.FALSE):
            self.advance()
            return BooleanLiteral(token.type == TokenType.TRUE)
        
        # Null
        elif self.match(TokenType.NULL):
            self.advance()
            return StringLiteral("null")
        
        # Arrays
        elif self.match(TokenType.LBRACKET):
            return self.parse_array_flexible()
        
        # Dictionaries
        elif self.match(TokenType.LBRACE):
            return self.parse_dict_flexible()
        
        # Parentheses
        elif self.match(TokenType.LPAREN):
            return self.parse_paren_flexible()
        
        # Everything else becomes an identifier
        else:
            self.advance()
            return Identifier(str(token.value))
    
    def parse_array_flexible(self) -> Expression:
        """Parse array with maximum error tolerance"""
        self.advance()  # consume [
        elements = []
        
        while not self.match(TokenType.RBRACKET, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.RBRACKET):
                break
            
            try:
                expr = self.parse_expression_flexible()
                if expr:
                    elements.append(expr)
            except:
                pass
            
            self.skip_newlines()
            if self.match(TokenType.COMMA):
                self.advance()
            elif not self.match(TokenType.RBRACKET):
                if not self.match(TokenType.EOF):
                    self.advance()
        
        if self.match(TokenType.RBRACKET):
            self.advance()
        return ArrayLiteral(elements)
    
    def parse_dict_flexible(self) -> Expression:
        """Parse dictionary with maximum error tolerance"""
        self.advance()  # consume {
        pairs = []
        
        while not self.match(TokenType.RBRACE, TokenType.EOF):
            try:
                key = self.parse_expression_flexible()
                if self.match(TokenType.COLON):
                    self.advance()
                    value = self.parse_expression_flexible()
                    if key and value:
                        pairs.append((key, value))
            except:
                pass
            
            if self.match(TokenType.COMMA):
                self.advance()
            elif not self.match(TokenType.RBRACE):
                if not self.match(TokenType.EOF):
                    self.advance()
        
        if self.match(TokenType.RBRACE):
            self.advance()
        return DictionaryLiteral(pairs)
    
    def parse_paren_flexible(self) -> Expression:
        """Parse parentheses with maximum error tolerance"""
        self.advance()  # consume (
        
        if self.match(TokenType.RPAREN):
            self.advance()
            return ArrayLiteral([])
        
        try:
            expr = self.parse_expression_flexible()
            if self.match(TokenType.RPAREN):
                self.advance()
            return expr or IntegerLiteral(0)
        except:
            if self.match(TokenType.RPAREN):
                self.advance()
            return IntegerLiteral(0)
    
    def parse_argument_list(self) -> List[Expression]:
        """Parse function arguments with maximum flexibility"""
        args = []
        
        while not self.match(TokenType.RPAREN, TokenType.EOF):
            try:
                # Try to parse expression
                expr = self.parse_expression_flexible()
                if expr:
                    args.append(expr)
                
                # Handle separators flexibly
                if self.match(TokenType.COMMA):
                    self.advance()
                elif self.match(TokenType.RPAREN):
                    break
                elif self.match(TokenType.NEWLINE):
                    self.advance()  # Skip newlines in argument lists
                else:
                    # Try to continue parsing
                    if not self.match(TokenType.EOF):
                        self.advance()
            except:
                # Skip problematic tokens
                if not self.match(TokenType.RPAREN, TokenType.EOF):
                    self.advance()
                else:
                    break
        
        return args
    
    def parse_expression_flexible(self) -> Optional[Expression]:
        """Parse expression with maximum error tolerance"""
        try:
            return self.parse_expression()
        except:
            # Create a dummy expression from current token
            if self.current_token and not self.match(TokenType.RPAREN, TokenType.EOF, TokenType.COMMA):
                if self.match(TokenType.IDENTIFIER):
                    expr = Identifier(self.current_token.value)
                    self.advance()
                    return expr
                elif self.match(TokenType.INTEGER):
                    expr = IntegerLiteral(int(self.current_token.value))
                    self.advance()
                    return expr
                elif self.match(TokenType.STRING):
                    expr = StringLiteral(self.current_token.value)
                    self.advance()
                    return expr
                else:
                    # Convert any token to identifier
                    expr = Identifier(str(self.current_token.value))
                    self.advance()
                    return expr
            return None
    
    def parse_kwarg_list(self) -> List[tuple[str, Expression]]:
        """Parse keyword arguments"""
        kwargs = []
        
        if not self.match(TokenType.RPAREN):
            kwargs.append(self.parse_kwarg())
            
            while self.match(TokenType.COMMA):
                self.advance()
                if self.match(TokenType.RPAREN):
                    break
                kwargs.append(self.parse_kwarg())
        
        return kwargs
    
    def parse_kwarg(self) -> tuple[str, Expression]:
        """Parse: name = value"""
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.ASSIGN)
        value = self.parse_expression()
        return (name, value)
    
    def parse_assignment_or_expression(self) -> Statement:
        """Parse assignment or expression statement"""
        try:
            # Handle keyword identifiers in assignments
            if self.is_keyword_as_identifier() and self.peek() and self.peek().type == TokenType.ASSIGN:
                name = self.current_token.value
                self.advance()
                self.consume(TokenType.ASSIGN)
                value = self.parse_expression()
                return Assignment(Identifier(name), value)
            
            expr = self.parse_expression()
            
            # Check for assignment
            if self.match(TokenType.ASSIGN):
                self.advance()
                value = self.parse_expression()
                # Optional semicolon
                if self.match(TokenType.SEMICOLON):
                    self.advance()
                return Assignment(expr, value)
            
            # Optional semicolon for expression statements
            if self.match(TokenType.SEMICOLON):
                self.advance()
            
            # Expression statement
            return ExpressionStatement(expr)
        except ParseError:
            # If expression parsing fails, try to recover by skipping to next line
            while not self.match(TokenType.NEWLINE, TokenType.EOF):
                self.advance()
            return None
    
    def parse_if_statement(self) -> IfStatement:
        """Parse if statement with else if support"""
        self.consume(TokenType.IF)
        condition = self.parse_expression()
        
        # Handle both : and { syntax flexibly
        if self.match(TokenType.COLON):
            self.advance()
        elif self.match(TokenType.LBRACE):
            pass  # Will be handled in parse_block
        
        then_body = self.parse_block()
        
        else_body = None
        # Handle else if chain
        if self.match(TokenType.ELSE):
            self.advance()
            
            # Check for "else if"
            if self.match(TokenType.IF):
                # Recursively parse the else if as a nested if statement
                else_body = [self.parse_if_statement()]
            else:
                # Regular else block
                if self.match(TokenType.COLON):
                    self.advance()
                elif self.match(TokenType.LBRACE):
                    pass
                else_body = self.parse_block()
        
        return IfStatement(condition, then_body, else_body)
    
    def parse_while_statement(self) -> WhileStatement:
        """Parse while statement"""
        self.consume(TokenType.WHILE)
        condition = self.parse_expression()
        
        # Colon is optional if using braces
        if self.match(TokenType.COLON):
            self.advance()
        elif not self.match(TokenType.LBRACE):
            raise ParseError("Expected ':' or '{'", self.current_token)
        
        # Parse while loop body with limited scope
        body = self.parse_limited_block()
        
        return WhileStatement(condition, body)
    
    def parse_limited_block(self) -> List[Statement]:
        """Parse a limited block for control structures"""
        # Use the improved block parsing
        return self.parse_block()
    
    def parse_for_statement(self) -> ForStatement:
        """Parse for statement"""
        self.consume(TokenType.FOR)
        variable = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.IN)
        iterable = self.parse_expression()
        
        # Colon is optional if using braces
        if self.match(TokenType.COLON):
            self.advance()
        elif not self.match(TokenType.LBRACE):
            raise ParseError("Expected ':' or '{'", self.current_token)
        
        body = self.parse_block()
        
        return ForStatement(variable, iterable, body)
    
    def parse_return_statement(self) -> ReturnStatement:
        """Parse return statement"""
        self.consume(TokenType.RETURN)
        
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF):
            value = self.parse_expression()
        
        return ReturnStatement(value)
    
    def parse_break_statement(self) -> BreakStatement:
        """Parse break statement"""
        self.consume(TokenType.BREAK)
        return BreakStatement()
    
    def parse_continue_statement(self) -> ContinueStatement:
        """Parse continue statement"""
        self.consume(TokenType.CONTINUE)
        return ContinueStatement()
    
    def parse_try_statement(self) -> TryStatement:
        """Parse try/except/finally statement"""
        self.consume(TokenType.TRY)
        
        if self.match(TokenType.COLON):
            self.advance()
        elif not self.match(TokenType.LBRACE):
            raise ParseError("Expected ':' or '{'", self.current_token)
        
        try_body = self.parse_block()
        
        except_clauses = []
        while self.match(TokenType.EXCEPT):
            self.advance()
            
            exception_type = None
            variable_name = None
            
            # Check if there's an exception type specified
            if (self.match(TokenType.IDENTIFIER) and 
                not self.match(TokenType.COLON) and 
                not self.match(TokenType.LBRACE)):
                exception_type = self.advance().value
                if self.match(TokenType.AS):
                    self.advance()
                    variable_name = self.consume(TokenType.IDENTIFIER).value
            
            if self.match(TokenType.COLON):
                self.advance()
            elif not self.match(TokenType.LBRACE):
                raise ParseError("Expected ':' or '{'", self.current_token)
            
            except_body = self.parse_block()
            except_clauses.append(ExceptClause(exception_type, variable_name, except_body))
        
        finally_body = None
        if self.match(TokenType.FINALLY):
            self.advance()
            if self.match(TokenType.COLON):
                self.advance()
            elif not self.match(TokenType.LBRACE):
                raise ParseError("Expected ':' or '{'", self.current_token)
            finally_body = self.parse_block()
        
        return TryStatement(try_body, except_clauses, finally_body)
    
    def parse_raise_statement(self) -> RaiseStatement:
        """Parse raise statement"""
        self.consume(TokenType.RAISE)
        
        exception = None
        if not self.match(TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF):
            exception = self.parse_expression()
        
        return RaiseStatement(exception)
    
    def parse_layer_statement(self) -> Statement:
        """Parse layer declaration: layer name = LayerType(args)"""
        self.consume(TokenType.LAYER)
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.ASSIGN)
        
        layer_type = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.LPAREN)
        args = self.parse_argument_list()
        self.consume(TokenType.RPAREN)
        
        activation = None
        if self.match(TokenType.COMMA):
            self.advance()
            activation = self.consume(TokenType.STRING).value.strip('"\'')
        
        layer_decl = LayerDeclaration(name, layer_type, args, activation)
        return ExpressionStatement(FunctionCall("define_layer", [StringLiteral(name), StringLiteral(layer_type)] + args))
    
    def parse_optimizer_statement(self) -> Statement:
        """Parse optimizer declaration: optimizer name = OptimizerType(args)"""
        self.consume(TokenType.OPTIMIZER)
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.ASSIGN)
        
        opt_type = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.LPAREN)
        args = self.parse_argument_list()
        self.consume(TokenType.RPAREN)
        
        return ExpressionStatement(FunctionCall("define_optimizer", [StringLiteral(name), StringLiteral(opt_type)] + args))
    
    def parse_import_statement(self) -> ImportStatement:
        """Parse import statement"""
        self.consume(TokenType.IMPORT)
        module_path = [self.consume(TokenType.IDENTIFIER).value]
        
        while self.match(TokenType.DOT):
            self.advance()
            module_path.append(self.consume(TokenType.IDENTIFIER).value)
        
        alias = None
        if self.match(TokenType.AS):
            self.advance()
            alias = self.consume(TokenType.IDENTIFIER).value
        
        return ImportStatement(module_path, alias)
    
    def parse_from_import_statement(self) -> FromImportStatement:
        """Parse from import statement"""
        self.consume(TokenType.FROM)
        module_path = [self.consume(TokenType.IDENTIFIER).value]
        
        while self.match(TokenType.DOT):
            self.advance()
            module_path.append(self.consume(TokenType.IDENTIFIER).value)
        
        self.consume(TokenType.IMPORT)
        names = [self.consume(TokenType.IDENTIFIER).value]
        
        while self.match(TokenType.COMMA):
            self.advance()
            names.append(self.consume(TokenType.IDENTIFIER).value)
        
        return FromImportStatement(module_path, names)
    
    def parse_type_definition(self) -> TypeDefinition:
        """Parse type definition"""
        self.consume(TokenType.TYPE)
        name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.LBRACE)
        
        fields = []
        while not self.match(TokenType.RBRACE):
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.COLON)
            field_type = self.parse_type()
            fields.append(FieldDeclaration(field_name, field_type))
            
            if self.match(TokenType.SEMICOLON):
                self.advance()
        
        self.consume(TokenType.RBRACE)
        return TypeDefinition(name, fields)
    
    def parse_optimize_call(self) -> OptimizeCall:
        """Parse optimize(loss) call"""
        self.consume(TokenType.OPTIMIZE)
        self.consume(TokenType.LPAREN)
        loss = self.parse_expression()
        self.consume(TokenType.RPAREN)
        
        return OptimizeCall(loss)
    
    def parse_log_call(self) -> LogCall:
        """Parse log(message) call"""
        self.consume(TokenType.LOG)
        self.consume(TokenType.LPAREN)
        message = self.parse_expression()
        
        values = []
        while self.match(TokenType.COMMA):
            self.advance()
            values.append(self.parse_expression())
        
        self.consume(TokenType.RPAREN)
        return LogCall(message, values)
    
    def parse_python_block(self) -> Statement:
        """Parse python: { ... } block - raw Python code execution"""
        self.consume(TokenType.IDENTIFIER)  # consume 'python'
        self.consume(TokenType.COLON)
        self.consume(TokenType.LBRACE)
        
        # Collect all tokens until matching closing brace
        python_code = ""
        brace_count = 1
        
        while brace_count > 0 and not self.match(TokenType.EOF):
            if self.match(TokenType.LBRACE):
                brace_count += 1
                python_code += "{"
            elif self.match(TokenType.RBRACE):
                brace_count -= 1
                if brace_count > 0:
                    python_code += "}"
            else:
                # Add the token value with appropriate spacing
                token_value = self.current_token.value
                if self.current_token.type == TokenType.NEWLINE:
                    python_code += "\n"
                elif self.current_token.type == TokenType.STRING:
                    python_code += f'"{token_value}"' if not (token_value.startswith('"') or token_value.startswith("'")) else token_value
                else:
                    python_code += token_value + " "
            
            self.advance()
        
        # Return as an expression statement that executes Python code
        return ExpressionStatement(FunctionCall("exec", [StringLiteral(python_code.strip())], []))

def parse(source: str) -> Program:
    """Convenience function to parse Aqualua source code"""
    tokens = tokenize(source)
    parser = AqualuaParser(tokens)
    return parser.parse()

if __name__ == "__main__":
    # Test the parser
    test_code = '''
    model MLP {
        l1: Linear(784, 256)
        l2: Linear(256, 10)
        
        fn forward(x: Tensor[B, 784]) -> Tensor[B, 10] {
            return l2(relu(l1(x)))
        }
    }
    
    let dataset = MNIST("data").batch(32)
    let model = MLP()
    
    train model m using Adam(lr=1e-3) on dataset for 5 epochs:
        step(batch b) {
            let out = m(b.x)
            let loss = cross_entropy(out, b.y)
            optimize(loss)
            log(loss)
        }
    '''
    
    try:
        ast = parse(test_code)
        print(print_ast(ast))
    except ParseError as e:
        print(f"Parse error: {e}")