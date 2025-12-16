"""
Aqualua AST (Abstract Syntax Tree) Nodes
Defines all the node types for the Aqualua language parse tree
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Any
from enum import Enum

# Base AST Node
class ASTNode(ABC):
    pass

# Expressions
class Expression(ASTNode):
    pass

class Statement(ASTNode):
    pass

# Literals
@dataclass
class IntegerLiteral(Expression):
    value: int

@dataclass
class FloatLiteral(Expression):
    value: float

@dataclass
class StringLiteral(Expression):
    value: str

@dataclass
class BooleanLiteral(Expression):
    value: bool

@dataclass
class ListLiteral(Expression):
    elements: List[Expression]

# Alias for compatibility
ArrayLiteral = ListLiteral

@dataclass
class DictionaryLiteral(Expression):
    pairs: List[tuple[Expression, Expression]]

# Identifiers
@dataclass
class Identifier(Expression):
    name: str

# Binary Operations
class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    MATMUL = "@"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    AND = "and"
    OR = "or"

@dataclass
class BinaryExpression(Expression):
    left: Expression
    operator: BinaryOp
    right: Expression

# Unary Operations
class UnaryOp(Enum):
    NEG = "-"
    POS = "+"
    NOT = "!"

@dataclass
class UnaryExpression(Expression):
    operator: UnaryOp
    operand: Expression

# Function Calls
@dataclass
class FunctionCall(Expression):
    name: str
    args: List[Expression]
    kwargs: List[tuple[str, Expression]]  # (name, value) pairs

# Indexing
@dataclass
class IndexExpression(Expression):
    object: Expression
    indices: List[Expression]

# Types
class Type(ASTNode):
    pass

@dataclass
class PrimitiveType(Type):
    name: str  # i32, f32, bool, str, etc.

@dataclass
class TensorType(Type):
    shape: List[Union[int, str]]  # dimensions, can be integers or symbolic names
    dtype: str

@dataclass
class ModelType(Type):
    name: str

# Variable Declarations
@dataclass
class VariableDeclaration(Statement):
    name: str
    type_annotation: Optional[Type]
    value: Expression
    is_const: bool = False

# Assignment
@dataclass
class Assignment(Statement):
    target: Expression  # Can be identifier or index expression
    value: Expression

# Function Definitions
@dataclass
class Parameter:
    name: str
    type_annotation: Type

@dataclass
class FunctionDefinition(Statement):
    name: str
    parameters: List[Parameter]
    return_type: Optional[Type]
    body: List[Statement]

# Model Definitions
@dataclass
class LayerDeclaration:
    name: str
    layer_type: str
    args: List[Expression]

@dataclass
class ModelDefinition(Statement):
    name: str
    layers: List[LayerDeclaration]
    methods: List[FunctionDefinition]

# Control Flow
@dataclass
class IfStatement(Statement):
    condition: Expression
    then_body: List[Statement]
    else_body: Optional[List[Statement]] = None

@dataclass
class WhileStatement(Statement):
    condition: Expression
    body: List[Statement]

@dataclass
class ForStatement(Statement):
    variable: str
    iterable: Expression
    body: List[Statement]

@dataclass
class ReturnStatement(Statement):
    value: Optional[Expression] = None

@dataclass
class BreakStatement(Statement):
    pass

@dataclass
class ContinueStatement(Statement):
    pass

@dataclass
class TryStatement(Statement):
    try_body: List[Statement]
    except_clauses: List['ExceptClause']
    finally_body: Optional[List[Statement]] = None

@dataclass
class ExceptClause:
    exception_type: Optional[str]
    variable_name: Optional[str]
    body: List[Statement]

@dataclass
class RaiseStatement(Statement):
    exception: Optional[Expression] = None

# AI-Specific AST Nodes
@dataclass
class ModelDefinition(Statement):
    name: str
    layers: List['LayerDeclaration']
    methods: List[FunctionDefinition]

@dataclass
class LayerDeclaration:
    name: str
    layer_type: str
    args: List[Expression]
    activation: Optional[str] = None

@dataclass
class TrainStatement(Statement):
    model: Expression
    optimizer: Expression
    dataset: Expression
    epochs: Expression
    loss_function: str

@dataclass
class TensorLiteral(Expression):
    shape: List[Expression]
    dtype: str
    data: Optional[List[Expression]] = None

@dataclass
class ModelCall(Expression):
    model: Expression
    input_data: Expression

# Training DSL
@dataclass
class OptimizerExpression(Expression):
    name: str
    kwargs: List[tuple[str, Expression]]

@dataclass
class StepBlock:
    parameter: str  # batch parameter name
    body: List[Statement]

@dataclass
class TrainStatement(Statement):
    model_name: str
    optimizer: Optional[OptimizerExpression]
    dataset: Expression
    epochs: Expression
    step_block: StepBlock

# Import Statements
@dataclass
class ImportStatement(Statement):
    module_path: List[str]
    alias: Optional[str] = None

@dataclass
class FromImportStatement(Statement):
    module_path: List[str]
    names: List[str]

# Type Definitions
@dataclass
class FieldDeclaration:
    name: str
    type_annotation: Type

@dataclass
class TypeDefinition(Statement):
    name: str
    fields: List[FieldDeclaration]

# Expression Statement (for expressions used as statements)
@dataclass
class ExpressionStatement(Statement):
    expression: Expression

# Program (top-level)
@dataclass
class Program(ASTNode):
    statements: List[Statement]

# Tensor Constructor
@dataclass
class TensorConstructor(Expression):
    shape: List[Expression]
    dtype: Optional[str] = None
    values: Optional[List[Expression]] = None

# Model Constructor
@dataclass
class ModelConstructor(Expression):
    model_type: str
    args: List[Expression]

# Special ML Operations
@dataclass
class OptimizeCall(Statement):
    loss: Expression

@dataclass
class LogCall(Statement):
    message: Expression
    values: List[Expression] = None

# Device Specification
@dataclass
class DeviceExpression(Expression):
    device_type: str  # "cpu", "gpu", "auto"

# Dataset Operations
@dataclass
class DatasetOperation(Expression):
    dataset: Expression
    operation: str  # "batch", "shuffle", "map", etc.
    args: List[Expression]

# Visitor Pattern for AST Traversal
class ASTVisitor(ABC):
    @abstractmethod
    def visit(self, node: ASTNode) -> Any:
        pass
    
    def visit_Program(self, node: Program) -> Any:
        return [self.visit(stmt) for stmt in node.statements]
    
    def visit_IntegerLiteral(self, node: IntegerLiteral) -> Any:
        return node.value
    
    def visit_FloatLiteral(self, node: FloatLiteral) -> Any:
        return node.value
    
    def visit_StringLiteral(self, node: StringLiteral) -> Any:
        return node.value
    
    def visit_BooleanLiteral(self, node: BooleanLiteral) -> Any:
        return node.value
    
    def visit_Identifier(self, node: Identifier) -> Any:
        return node.name
    
    def visit_BinaryExpression(self, node: BinaryExpression) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return (node.operator, left, right)
    
    def visit_UnaryExpression(self, node: UnaryExpression) -> Any:
        operand = self.visit(node.operand)
        return (node.operator, operand)
    
    def visit_FunctionCall(self, node: FunctionCall) -> Any:
        args = [self.visit(arg) for arg in node.args]
        kwargs = [(name, self.visit(value)) for name, value in node.kwargs]
        return (node.name, args, kwargs)

# Pretty Printer for AST
class ASTPrinter(ASTVisitor):
    def __init__(self):
        self.indent_level = 0
    
    def indent(self):
        return "  " * self.indent_level
    
    def visit(self, node: ASTNode) -> str:
        method_name = f"visit_{type(node).__name__}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(node)
        else:
            return f"{self.indent()}{type(node).__name__}({node})"
    
    def visit_Program(self, node: Program) -> str:
        self.indent_level += 1
        statements = "\n".join(self.visit(stmt) for stmt in node.statements)
        self.indent_level -= 1
        return f"Program:\n{statements}"
    
    def visit_ModelDefinition(self, node: ModelDefinition) -> str:
        result = f"{self.indent()}Model {node.name}:\n"
        self.indent_level += 1
        
        for layer in node.layers:
            result += f"{self.indent()}{layer.name}: {layer.layer_type}\n"
        
        for method in node.methods:
            result += self.visit(method) + "\n"
        
        self.indent_level -= 1
        return result
    
    def visit_FunctionDefinition(self, node: FunctionDefinition) -> str:
        params = ", ".join(f"{p.name}: {p.type_annotation}" for p in node.parameters)
        return_type = f" -> {node.return_type}" if node.return_type else ""
        
        result = f"{self.indent()}fn {node.name}({params}){return_type}:\n"
        self.indent_level += 1
        
        for stmt in node.body:
            result += self.visit(stmt) + "\n"
        
        self.indent_level -= 1
        return result

def print_ast(node: ASTNode) -> str:
    """Convenience function to pretty print an AST"""
    printer = ASTPrinter()
    return printer.visit(node)