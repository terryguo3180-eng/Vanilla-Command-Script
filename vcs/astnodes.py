from __future__ import annotations

from typing import Any, TYPE_CHECKING

from vcs import lexer as lex
from vcs import utils

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
                        self.visit(item)
            elif isinstance(value, ASTNode):
                self.visit(value)


class Module(ASTNode):
    __slots__ = ("body", "scope")

    def __init__(self, body: list[FunctionDeclaration | Comment], **loc):
        super().__init__(**loc)
        self.body = body

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


class Comment(ASTNode):
    __slots__ = ("token",)

    def __init__(self, token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.token = token


# Declarations

class FunctionDeclaration(ASTNode):
    __slots__ = (
        "name_token",
        "lparen_token",
        "params",
        "rparen_token",
        "arrow_token",
        "return_type",
        "colon_token",
        "body",
        # Fields for semantic analyzer
        "signature",
        "scope",
    )

    def __init__(
        self,
        name_token: lex.TokenInfo,
        lparen_token: lex.TokenInfo,
        params: list[Parameter],
        rparen_token: lex.TokenInfo,
        arrow_token: lex.TokenInfo | None,
        return_type: Type | None,
        colon_token: lex.TokenInfo | None,
        body: Statement | None,
        **loc,
    ):
        super().__init__(**loc)
        self.name_token = name_token
        self.lparen_token = lparen_token
        self.params = params
        self.rparen_token = rparen_token
        self.arrow_token = arrow_token
        self.return_type = return_type
        self.colon_token = colon_token
        self.body = body

    def annotate_signature(self, signature: sem.FunctionTypeInfo):
        self.signature = signature

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


class Parameter(ASTNode):
    __slots__ = ("name_token", "colon_token", "type", "equal_token", "default")

    def __init__(
        self,
        name_token: lex.TokenInfo,
        colon_token: lex.TokenInfo | None,
        type: Type,
        equal_token: lex.TokenInfo | None,
        default: Expression | None,
        **loc,
    ):
        super().__init__(**loc)
        self.name_token = name_token
        self.colon_token = colon_token
        self.type = type
        self.equal_token = equal_token
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
    __slots__ = (
        "name_token",
        "colon_token",
        "type",
        "equal_token",
        "value",
    )

    def __init__(
        self,
        name_token: lex.TokenInfo,
        colon_token: lex.TokenInfo | None,
        type: Type,
        equal_token: lex.TokenInfo | None,
        value: Expression | None,
        **loc,
    ):
        super().__init__(**loc)
        self.name_token = name_token
        self.colon_token = colon_token
        self.type = type
        self.equal_token = equal_token
        self.value = value


class AssignStatement(Statement):
    __slots__ = ("target", "equal_token", "value")

    def __init__(
        self,
        target: LeftExpression,
        equal_token: lex.TokenInfo,
        value: Expression,
        **loc,
    ):
        super().__init__(**loc)
        self.target = target
        self.equal_token = equal_token
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
    __slots__ = ("return_token", "value")

    def __init__(self, return_token: lex.TokenInfo, value: Expression | None, **loc):
        super().__init__(**loc)
        self.return_token = return_token
        self.value = value


class BreakStatement(Statement):
    __slots__ = ("break_token",)

    def __init__(self, break_token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.break_token = break_token


class ContinueStatement(Statement):
    __slots__ = ("continue_token",)

    def __init__(self, continue_token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.continue_token = continue_token


class PassStatement(Statement):
    __slots__ = ("pass_token",)

    def __init__(self, pass_token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.pass_token = pass_token


class IfStatement(Statement):
    __slots__ = (
        "if_token",
        "test",
        "colon_token",
        "body",
        "else_token",
        "else_colon_token",
        "orelse",
    )

    def __init__(
        self,
        if_token: lex.TokenInfo,
        test: Expression,
        colon_token: lex.TokenInfo | None,
        body: Statement,
        else_token: lex.TokenInfo | None,
        else_colon_token: lex.TokenInfo | None,
        orelse: Statement | None,
        **loc,
    ):
        super().__init__(**loc)
        self.if_token = if_token
        self.test = test
        self.colon_token = colon_token
        self.body = body
        self.else_token = else_token
        self.else_colon_token = else_colon_token
        self.orelse = orelse


class WhileStatement(Statement):
    __slots__ = ("while_token", "test", "colon_token", "body")

    def __init__(
        self,
        while_token: lex.TokenInfo,
        test: Expression,
        colon_token: lex.TokenInfo | None,
        body: Statement,
        **loc,
    ):
        super().__init__(**loc)
        self.while_token = while_token
        self.test = test
        self.colon_token = colon_token
        self.body = body


class ForStatement(Statement):
    __slots__ = (
        "for_token",
        "lparen_token",
        "init_stmt",
        "semicolon_token1",
        "test",
        "semicolon_token2",
        "end_stmt",
        "rparen_token",
        "colon_token",
        "body",

        "scope",
    )

    def __init__(
        self,
        for_token: lex.TokenInfo,
        init_stmt: Statement,
        semicolon_token1: lex.TokenInfo,
        test: Expression,
        semicolon_token2: lex.TokenInfo,
        end_stmt: Statement,
        colon_token: lex.TokenInfo,
        body: Statement,
        lparen_token: lex.TokenInfo | None,
        rparen_token: lex.TokenInfo | None,
        **loc,
    ):
        super().__init__(**loc)
        self.for_token = for_token
        self.lparen_token = lparen_token
        self.init_stmt = init_stmt
        self.semicolon_token1 = semicolon_token1
        self.test = test
        self.semicolon_token2 = semicolon_token2
        self.end_stmt = end_stmt
        self.rparen_token = rparen_token
        self.colon_token = colon_token
        self.body = body

    def annotate_scope(self, scope: sem.Scope):
        self.scope = scope


# Expressions


class Expression(ASTNode):
    __slots__ = ("type_info",)

    def annotate_type(self, type: sem.TypeInfo):
        self.type_info = type


class LeftExpression(Expression): ...


class IfExpression(Expression):
    __slots__ = ("test", "if_token", "body", "else_token", "orelse")

    def __init__(
        self,
        test: Expression,
        if_token: lex.TokenInfo,
        body: Expression,
        else_token: lex.TokenInfo,
        orelse: Expression,
        **loc,
    ):
        super().__init__(**loc)
        self.test = test
        self.if_token = if_token
        self.body = body
        self.else_token = else_token
        self.orelse = orelse


class UnaryExpression(Expression):
    __slots__ = ("operand", "op")

    def __init__(self, operand: Expression, op: UnaryOp, **loc):
        super().__init__(**loc)
        self.operand = operand
        self.op = op


class BinaryExpression(Expression):
    __slots__ = ("left", "op", "right")

    def __init__(self, left: Expression, op: BinaryOp, right: Expression, **loc):
        super().__init__(**loc)
        self.left = left
        self.op = op
        self.right = right


class CallExpression(Expression):
    __slots__ = ("callee", "lparen_token", "args", "rparen_token")

    def __init__(
        self,
        callee: Expression,
        lparen_token: lex.TokenInfo | None,
        args: list[Argument],
        rparen_token: lex.TokenInfo | None,
        **loc,
    ):
        super().__init__(**loc)
        self.callee = callee
        self.lparen_token = lparen_token
        self.args = args
        self.rparen_token = rparen_token


class Argument(ASTNode):
    __slots__ = ("value",)
    value: Expression


class PositionalArgument(Argument):
    def __init__(self, value: Expression, **loc):
        super().__init__(**loc)
        self.value = value


class KeywordArgument(Argument):
    __slots__ = ("name_token", "equal_token")

    def __init__(
        self, name_token: lex.TokenInfo, equal_token: lex.TokenInfo, value: Expression, **loc
    ):
        super().__init__(**loc)
        self.name_token = name_token
        self.equal_token = equal_token
        self.value = value


class Constant(Expression):
    __slots__ = ("value",)

    def __init__(self, value: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.value = value


class Identifier(LeftExpression):
    __slots__ = ("token", "context")

    def __init__(self, token: lex.TokenInfo, context: Context, **loc):
        super().__init__(**loc)
        self.token = token
        self.context = context


class Context(ASTNode): ...
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


class Type(ASTNode):
    __slots__ = ("token",)

    def __init__(self, token: lex.TokenInfo, **loc):
        super().__init__(**loc)
        self.token = token


class IntType(Type): ...
class BoolType(Type): ...
class FloatType(Type): ...
