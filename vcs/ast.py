from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING, Self

from vcs import utils
from vcs import lexer as lex

if TYPE_CHECKING:
    from vcs import semantic as sem


class ASTNode:
    __slots__ = ("filename", "lineno", "column", "end_lineno", "end_column")

    def __init__(
        self,
        filename: str = "<unknown>",
        lineno: int = 0,
        column: int = 0,
        end_lineno: int = 0,
        end_column: int = 0,
    ):
        self.filename = filename
        self.lineno = lineno
        self.column = column
        self.end_lineno = end_lineno
        self.end_column = end_column

    def __repr__(self) -> str:
        return utils.dump_astnode(self, indent=None)
    
    def __str__(self) -> str:
        return utils.dump_astnode(self)
    
    def location(self) -> dict[str, str | int]:
        return {
            "filename": self.filename,
            "lineno": self.lineno,
            "column": self.column,
            "end_lineno": self.end_lineno,
            "end_column": self.end_column,
        }
    
    def copy(self) -> Self:
        return deepcopy(self)


class ASTNodeVisitor:
    def visit(self, node, *args, **kwargs):
        method = "visit_" + type(node).__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: ASTNode, *args, **kwargs) -> Any:
        for _, value in utils.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        self.visit(item, *args, **kwargs)
            elif isinstance(value, ASTNode):
                self.visit(value, *args, **kwargs)


class ASTNodeTransformer(ASTNodeVisitor):
    def generic_visit(self, node, *args, **kwargs):
        for field, old_value in utils.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ASTNode):
                        value = self.visit(value, *args, **kwargs)
                        if value is None:
                            continue
                        elif not isinstance(value, ASTNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ASTNode):
                new_node = self.visit(old_value, *args, **kwargs)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class Module(ASTNode):
    __slots__ = ("body", "scope")

    def __init__(self, body: list[FunctionDeclaration | Comment], **loc):
        super().__init__(**loc)
        self.body = body

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


class Comment(ASTNode):
    __slots__ = ("value",)

    def __init__(self, value: str, **loc):
        super().__init__(**loc)
        self.value = value


# Declarations

class FunctionDeclaration(ASTNode):
    __slots__ = ("name", "params", "return_type", "body", "name_token", "signature", "scope")

    def __init__(
        self,
        name: str,
        params: list[Parameter],
        return_type: Type | None,
        body: Statement | None,
        name_token: lex.TokenInfo,
        **loc,
    ):
        super().__init__(**loc)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body
        self.name_token = name_token

    def annotate_signature(self, signature: sem.FunctionTypeInfo):
        self.signature = signature

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


class Parameter(ASTNode):
    __slots__ = ("name", "type", "default")

    def __init__(
        self,
        name: str,
        type: Type,
        default: Expression | None,
        **loc,
    ):
        super().__init__(**loc)
        self.name = name
        self.type = type
        self.default = default


# Statements

class Statement(ASTNode): ...


class Block(Statement):
    __slots__ = ("body", "scope")

    def __init__(
        self,
        body: list[Statement],
        **loc,
    ):
        super().__init__(**loc)
        self.body = body
    
    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


class ExpressionStatement(Statement):
    __slots__ = ("expression",)

    def __init__(self, expression: Expression, **loc):
        super().__init__(**loc)
        self.expression = expression


class VariableDeclarationStatement(Statement):
    __slots__ = ("name", "type", "value", "name_token")

    def __init__(
        self,
        name: str,
        type: Type,
        value: Expression | None,
        name_token: lex.TokenInfo,
        **loc
    ):
        super().__init__(**loc)
        self.name = name
        self.type = type
        self.value = value
        self.name_token = name_token


class AssignStatement(Statement):
    __slots__ = ("target", "value")

    def __init__(
        self,
        target: LeftExpression,
        value: Expression,
        **loc,
    ):
        super().__init__(**loc)
        self.target = target
        self.value = value


class AugAssignStatement(Statement):
    __slots__ = ("target", "op", "value")

    def __init__(
        self,
        target: LeftExpression,
        op: ArithmeticOp,
        value: Expression,
        **loc
    ):
        super().__init__(**loc)
        self.target = target
        self.op = op
        self.value = value


class ReturnStatement(Statement):
    __slots__ = ("value",)

    def __init__(self, value: Expression | None, **loc):
        super().__init__(**loc)
        self.value = value


class BreakStatement(Statement): ...

class ContinueStatement(Statement): ...

class PassStatement(Statement): ...


class IfStatement(Statement):
    __slots__ = ("test", "body", "orelse")

    def __init__(self, test: Expression, body: Statement, orelse: Statement | None, **loc):
        super().__init__(**loc)
        self.test = test
        self.body = body
        self.orelse = orelse


class WhileStatement(Statement):
    __slots__ = ("test", "body")

    def __init__(self, test: Expression | None, body: Statement, **loc,):
        super().__init__(**loc)
        self.test = test
        self.body = body


class ForStatement(Statement):
    __slots__ = ("init_stmt", "test", "end_stmt", "body", "scope")

    def __init__(
        self,
        init_stmt: Statement | None,
        test: Expression | None,
        end_stmt: Statement | None,
        body: Statement,
        **loc,
    ):
        super().__init__(**loc)
        self.init_stmt = init_stmt
        self.test = test
        self.end_stmt = end_stmt
        self.body = body

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


# Expressions


class Expression(ASTNode):
    __slots__ = ("type_info", "evaluation_scope")

    def annotate_type(self, type: sem.TypeInfo):
        self.type_info = type
        self.evaluation_scope = None

    def annotate_evaluation_scope(self, scope: sem.Scope):
        self.evaluation_scope = scope


class LeftExpression(Expression): ...


class IfExpression(Expression):
    __slots__ = ("test", "body", "orelse")

    def __init__(
        self,
        test: Expression,
        body: Expression,
        orelse: Expression,
        **loc,
    ):
        super().__init__(**loc)
        self.test = test
        self.body = body
        self.orelse = orelse


class UnaryExpression(Expression):
    __slots__ = ("operand", "op")

    def __init__(self, operand: Expression, op: UnaryOp, **loc):
        super().__init__(**loc)
        self.operand = operand
        self.op = op


class BinaryExpression(Expression):
    __slots__ = ("lhs", "op", "rhs")

    def __init__(self, lhs: Expression, op: BinaryOp, rhs: Expression, **loc):
        super().__init__(**loc)
        self.lhs = lhs
        self.op = op
        self.rhs = rhs


class CallExpression(Expression):
    __slots__ = ("callee", "args")

    def __init__(
        self,
        callee: Expression,
        args: list[Argument],
        **loc,
    ):
        super().__init__(**loc)
        self.callee = callee
        self.args = args


class Argument(ASTNode):
    __slots__ = ("value",)
    value: Expression


class PositionalArgument(Argument):
    def __init__(self, value: Expression, **loc):
        super().__init__(**loc)
        self.value = value


class KeywordArgument(Argument):
    __slots__ = ("name",)

    def __init__(self, name: str, value: Expression, **loc):
        super().__init__(**loc)
        self.name = name
        self.value = value


class Constant(Expression):
    __slots__ = ("value", "type")

    def __init__(self, value: Any, type: Type, **loc):
        super().__init__(**loc)
        self.value = value
        self.type = type


class Identifier(LeftExpression):
    __slots__ = ("context",)

    def __init__(self, name: str, context: Context, **loc):
        super().__init__(**loc)
        self.name = name
        self.context = context


class Context(ASTNode, metaclass=utils.SingletonMeta): ...
class Store(Context): ...
class Load(Context): ...


class Operator(ASTNode):
    __slots__ = ("token",)

    def __init__(self, token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.token = token


class BinaryOp(Operator): ...

class ArithmeticOp(BinaryOp): ...
class AddOp(ArithmeticOp): ...
class SubOp(ArithmeticOp): ...
class MulOp(ArithmeticOp): ...
class DivOp(ArithmeticOp): ...
class FloorDivOp(ArithmeticOp): ...
class ModOp(ArithmeticOp): ...

class CompareOp(BinaryOp): ...

class EqOp(CompareOp): ...
class NeOp(CompareOp): ...
class GtOp(CompareOp): ...
class LtOp(CompareOp): ...
class GeOp(CompareOp): ...
class LeOp(CompareOp): ...

class BinaryBoolOp(BinaryOp): ...
class OrOp(BinaryBoolOp): ...
class AndOp(BinaryBoolOp): ...

class UnaryOp(Operator): ...
class PosOp(UnaryOp): ...
class NegOp(UnaryOp): ...
class NotOp(UnaryOp): ...


class Type(ASTNode): ...

class IntType(Type): ...
class BoolType(Type): ...
class FloatType(Type): ...
class StringType(Type): ...
