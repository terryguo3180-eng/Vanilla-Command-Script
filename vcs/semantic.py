from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast, overload

from vcs import ast
from vcs import errors as err
from vcs import constfold as cf
from vcs import utils


class TypeInfo:
    """
    Base class for all type information in the semantic analysis phase.
    Represents the type of a variable, expression, or function return value.
    """


class VoidTypeInfo(TypeInfo, metaclass=utils.SingletonMeta):
    """
    Represents the void type, used for functions that return no value.
    """
    
    def __str__(self):
        return "Void"
    
    def __eq__(self, other):
        return isinstance(other, VoidTypeInfo)


class IntTypeInfo(TypeInfo, metaclass=utils.SingletonMeta):
    """
    Represents the integer type (e.g., 42, -17, 0).
    """
    
    def __str__(self):
        return "Int"
    
    def __eq__(self, other):
        return isinstance(other, IntTypeInfo)


class BoolTypeInfo(TypeInfo, metaclass=utils.SingletonMeta):
    """
    Represents the boolean type (true/false).
    """
    
    def __str__(self):
        return "Bool"
    
    def __eq__(self, other):
        return isinstance(other, BoolTypeInfo)


class FloatTypeInfo(TypeInfo, metaclass=utils.SingletonMeta):
    """
    Represents the floating-point type (e.g., 3.14, -2.5).
    """
    
    def __str__(self):
        return "Float"
    
    def __eq__(self, other):
        return isinstance(other, FloatTypeInfo)


class ErrorTypeInfo(TypeInfo, metaclass=utils.SingletonMeta):
    """
    Represents an error type, used when type checking fails.
    Propagates through the AST to avoid cascading errors.
    """


class ParameterInfo:
    """
    Stores information about a function parameter including its name,
    type, and optional default value expression.
    """
    
    def __init__(
        self, name: str,
        type_info: TypeInfo,
        default_node: ast.Expression | None = None
    ):
        self.name = name
        self.type_info = type_info
        self.default_node = default_node


class FunctionTypeInfo(TypeInfo):
    """
    Represents the type signature of a function including parameters and return type.
    """
    
    def __init__(
        self, param_infos: list[ParameterInfo], return_type: TypeInfo
    ):
        self.param_infos = param_infos
        self.return_type = return_type


class SymbolKind(Enum):
    """
    Enumeration of the different kinds of symbols that can be stored.
    """
    VARIABLE = auto()  # Local or global variable
    PARAMETER = auto()  # Function parameter
    FUNCTION = auto()  # Function declaration


@dataclass(frozen=True)
class Symbol:
    """
    Represents a named symbol in the symbol table with its kind and type.
    """
    
    name: str
    kind: SymbolKind
    type_info: TypeInfo

    def __hash__(self):
        return id(self)


class Scope:
    """
    Represents a lexical scope (e.g., function body, block statement).
    Scopes form a hierarchy through the enclosing reference.
    """
    
    def __init__(self, enclosing: Scope | None = None):
        self.enclosing = enclosing  # Parent scope, or None for global scope
        self._symbols: dict[str, Symbol] = {}  # Symbols declared in this scope

    def declare(self, symbol: Symbol) -> None:
        self._symbols[symbol.name] = symbol

    def lookup_local(self, name: str) -> Symbol | None:
        return self._symbols.get(name)

    def get_symbols(self) -> list[Symbol]:
        return list(self._symbols.values())

    def lookup(self, name: str) -> Symbol | None:
        scope = self
        while scope is not None:
            sym = scope.lookup_local(name)
            if sym is not None:
                return sym
            scope = scope.enclosing
        return None
    
    def __repr__(self):
        return f"Scope(symbols={list(self._symbols.keys())}, enclosing={self.enclosing is not None})"


class SymbolTable:
    """
    Manages scopes and symbol lookup for the entire program.
    Tracks the current scope and provides methods for entering/exiting scopes.
    """
    
    def __init__(self):
        self.global_scope = Scope(enclosing=None)
        self.cur_scope = self.global_scope

    def enter_scope(self) -> Scope:
        new_scope = Scope(enclosing=self.cur_scope)
        self.cur_scope = new_scope
        return new_scope

    def exit_scope(self) -> None:
        if self.cur_scope.enclosing is not None:
            self.cur_scope = self.cur_scope.enclosing
        else:
            raise RuntimeError("Cannot exit global scope")
    
    @contextmanager
    def with_scope(self, scope: Scope):
        prev = self.cur_scope
        self.set_scope(scope)
        yield scope
        self.set_scope(prev)

    def set_scope(self, scope: Scope):
        self.cur_scope = scope

    def declare_var(self, name: str, type_info: TypeInfo) -> None:
        sym = Symbol(name, SymbolKind.VARIABLE, type_info)
        self.cur_scope.declare(sym)

    def declare_param(self, name: str, type_info: TypeInfo) -> None:
        sym = Symbol(name, SymbolKind.PARAMETER, type_info)
        self.cur_scope.declare(sym)

    def declare_func(self, name: str, signature: FunctionTypeInfo) -> None:
        sym = Symbol(name, SymbolKind.FUNCTION, signature)
        self.cur_scope.declare(sym)

    def lookup(self, name: str) -> Symbol | None:
        return self.cur_scope.lookup(name)

    def lookup_func_signature(self, name: str) -> FunctionTypeInfo | None:
        sym = self.lookup(name)
        if sym and sym.kind == SymbolKind.FUNCTION:
            return cast(FunctionTypeInfo, sym.type_info)
        return None

    def is_declared_in_current_scope(self, name: str) -> bool:
        return self.cur_scope.lookup_local(name) is not None

    def get_locals(self) -> list[Symbol]:
        result = []
        scope = self.cur_scope
        while scope and scope is not self.global_scope:
            result.extend(scope.get_symbols())
            scope = scope.enclosing
        return result

    def register_name(self, name: str, type_info: TypeInfo):
        self.declare_var(name, type_info)


class SemanticAnalyzer(ast.ASTNodeVisitor):
    """
    Performs semantic analysis and type checking on the AST.
    Verifies type correctness, symbol resolution, and semantic rules.
    """
    
    def __init__(self, constfolder: cf.ConstantFolder, errors: err.ErrorCollector):
        self.constfolder = constfolder
        self.errors = errors
        self.get_error_info_on = constfolder.get_error_info_on  # Factory for error location info

        self.symtab = SymbolTable()  # Symbol table for scope management

        self.current_return_type: TypeInfo  # Type expected for return statements
        self.inside_loop = False  # Track if we're inside a loop (for break/continue)

        # Map call expressions to their argument bindings for later analysis
        self.call_param_bindings: dict[ast.CallExpression, dict[str, ast.Expression]] = {}

    def report(self, error: err.CompilerError):
        self.errors.add(error)

    def analyze(self) -> ast.Module | None:
        tree = self.constfolder.fold()
        self.visit(tree)
        if not self.errors.ok():
            return None
        return tree

    @overload
    def visit(self, node: ast.Module) -> None: ...
    @overload
    def visit(self, node: ast.FunctionDeclaration) -> None: ...
    @overload
    def visit(self, node: ast.Comment) -> None: ...
    @overload
    def visit(self, node: ast.Statement) -> None: ...
    @overload
    def visit(self, node: ast.Expression) -> None: ...
    @overload
    def visit(self, node: ast.Type) -> TypeInfo: ...

    def visit(self, node):
        return super().visit(node)

    def enter_scope(self):
        return self.symtab.enter_scope()
    
    def exit_scope(self):
        self.symtab.exit_scope()
    
    def with_scope(self, scope: Scope):
        return self.symtab.with_scope(scope)
    
    def register_name(self, name: str, type: TypeInfo):
        self.symtab.declare_var(name, type)

    def declare_func(self, name: str, params: list[ast.Parameter], return_type: TypeInfo):
        param_infos = []
        for param in params:
            declared_type = self.visit(param.type)
            param_infos.append(ParameterInfo(param.name, declared_type, param.default))
        
        signature = FunctionTypeInfo(param_infos, return_type)
        self.symtab.declare_func(name, signature)

    def declare_var(self, name: str, declared_type: TypeInfo, value: ast.Expression | None):
        self.register_name(name, declared_type)
        
        if isinstance(declared_type, ErrorTypeInfo):
            return  # Don't check if type is already erroneous
        
        if value is not None:
            self.visit(value)
            if isinstance(value.type_info, ErrorTypeInfo):
                return
            
            # Check that initializer type matches declared type
            if value.type_info != declared_type:
                self.report(err.UnassignableType(
                    self.get_error_info_on(value),
                    str(value.type_info), str(declared_type)
                ))

    def lookup_vartype(self, name: str) -> TypeInfo | None:
        sym = self.symtab.lookup(name)
        if sym is not None:
            return sym.type_info
        return None

    def check_assignment(self, target: ast.LeftExpression, value: ast.Expression) -> bool:
        # Check that a value can be assigned to a target (type compatibility)
        self.visit(target)
        self.visit(value)

        if target.type_info == value.type_info:
            return True

        if isinstance(target.type_info, ErrorTypeInfo) or isinstance(value.type_info, ErrorTypeInfo):
            return True

        if isinstance(value.type_info, IntTypeInfo) and isinstance(target.type_info, FloatTypeInfo):
            return True

        self.report(err.UnassignableType(
            self.get_error_info_on(value),
            str(value.type_info), str(target.type_info)
        ))
        return False


    def visit_Module(self, node: ast.Module):
        # Entry point
        scope = self.symtab.global_scope
        node.annotate_scope(scope)

        # First pass: collect all function signatures
        for sub in node.body:
            if isinstance(sub, ast.FunctionDeclaration):
                return_type = (
                    self.visit(sub.return_type)
                    if sub.return_type is not None
                    else VoidTypeInfo()
                )
                name = sub.name
                if self.symtab.lookup_func_signature(name) is not None:
                    self.report(err.FunctionDeclared(
                        self.get_error_info_on(sub.name_token), name
                    ))
                self.declare_func(name, sub.params, return_type)

        # Second pass: process function bodies
        for sub in node.body:
            self.visit(sub)

    def visit_FunctionDeclaration(self, node: ast.FunctionDeclaration):
        # Create scope, declare parameters, and type-check the function body
        prev_scope = self.symtab.cur_scope

        scope = self.enter_scope()
        node.annotate_scope(scope)

        name = node.name_token.value
        signature = self.symtab.lookup_func_signature(name)
        assert signature is not None
        node.annotate_signature(signature)

        # Set up the function's return type and declare parameters
        self.current_return_type = signature.return_type
        for param_info in signature.param_infos:
            self.declare_var(param_info.name, param_info.type_info, param_info.default_node)

            # Handling default expression
            default_node = param_info.default_node
            if default_node is not None:
                # Referencing previous arguments in the default expression is not allowed,
                # so we need to temporarily switch back to the previous scope
                with self.with_scope(prev_scope):
                    self.visit(default_node)
                    default_node.annotate_evaluation_scope(prev_scope)
        
        # Type-check the function body if it exists
        if node.body is not None:
            self.visit(node.body)

        self.exit_scope()
    
    def visit_Comment(self, node: ast.Comment):
        pass

    def visit_Block(self, node: ast.Block):
        # Process a block statement as a new lexical scope
        scope = self.enter_scope()
        node.annotate_scope(scope)
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()
    
    def visit_ExpressionStatement(self, node: ast.ExpressionStatement):
        # Process an expression statement (expression used as a statement)
        self.visit(node.expression)

    def visit_VariableDeclarationStatement(self, node: ast.VariableDeclarationStatement):
        # Process a variable declaration, checking for redeclaration and type compatibility
        name = node.name_token.value
        if self.symtab.is_declared_in_current_scope(name):
            self.report(err.VariableDeclared(
                self.get_error_info_on(node.name_token), name
            ))
        declared_type = self.visit(node.type)
        self.declare_var(name, declared_type, node.value)

    def visit_AssignStatement(self, node: ast.AssignStatement):
        # Process a regular assignment statement
        self.check_assignment(node.target, node.value)
    
    def visit_AugAssignStatement(self, node: ast.AugAssignStatement):
        # Process an augmented assignment (e.g., +=, -=, etc.)
        if self.check_assignment(node.target, node.value):
            lhs_type = node.target.type_info
            rhs_type = node.value.type_info

            lhs_int = isinstance(lhs_type, IntTypeInfo)
            lhs_float = isinstance(lhs_type, FloatTypeInfo)
            rhs_int = isinstance(rhs_type, IntTypeInfo)
            rhs_float = isinstance(rhs_type, FloatTypeInfo)

            if lhs_int and rhs_int:
                return
            
            if lhs_float and rhs_float:
                return
            
            if lhs_float and rhs_int:
                return
            
            self.report(err.InvalidAugmentedOpTypes(
                self.get_error_info_on(node),
                str(lhs_type), str(rhs_type), node.op.token.value
            ))
    
    def visit_PassStatement(self, node: ast.PassStatement):
        pass

    def visit_ReturnStatement(self, node: ast.ReturnStatement):
        # Process a return statement, checking type compatibility with function return type
        if node.value is None:
            value_type = VoidTypeInfo()
        else:
            self.visit(node.value)
            value_type = node.value.type_info
        
        if isinstance(self.current_return_type, ErrorTypeInfo):
            return
        if isinstance(value_type, ErrorTypeInfo):
            return

        if value_type != self.current_return_type:
            err_node = node.value or node
            self.report(err.UnassignableType(
                self.get_error_info_on(err_node),
                str(value_type), str(self.current_return_type)
            ))
    
    def _check_bool(self, node: ast.Expression):
        return isinstance(node.type_info, IntTypeInfo) or isinstance(node.type_info, BoolTypeInfo)

    def visit_IfStatement(self, node: ast.IfStatement):
        self.visit(node.test)  # Condition expression
        if not self._check_bool(node.test):
            self.report(err.InvalidConditionType(
                self.get_error_info_on(node.test), str(node.test.type_info)
            ))
        self.visit(node.body)  # Then branch
        if node.orelse:
            self.visit(node.orelse)  # Else branch (if present)
    
    def visit_WhileStatement(self, node: ast.WhileStatement):
        # Process a while loop, tracking loop context for break/continue
        if node.test is not None:
            self.visit(node.test)
            if not self._check_bool(node.test):
                self.report(err.InvalidConditionType(
                    self.get_error_info_on(node.test), str(node.test.type_info)
                ))
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)
        self.inside_loop = was_loop
    
    def visit_ForStatement(self, node: ast.ForStatement):
        # Process a for loop, which has its own scope for initialization
        scope = self.enter_scope()
        node.annotate_scope(scope)
        if node.init_stmt is not None:
            self.visit(node.init_stmt)  # Initialization statement
        if node.test is not None:
            self.visit(node.test)  # Loop condition
            if not self._check_bool(node.test):
                self.report(err.InvalidConditionType(
                    self.get_error_info_on(node.test), str(node.test.type_info)
                ))
        if node.end_stmt is not None:
            self.visit(node.end_stmt)  # End-of-iteration statement
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)  # Loop body
        self.inside_loop = was_loop
        self.exit_scope()

    def visit_BreakStatement(self, node: ast.BreakStatement):
        # Check that break is used only inside a loop
        if not self.inside_loop:
            self.report(err.BreakOutsideLoop(self.get_error_info_on(node)))
    
    def visit_ContinueStatement(self, node: ast.ContinueStatement):
        # Check that continue is used only inside a loop
        if not self.inside_loop:
            self.report(err.ContinueOutsideLoop(self.get_error_info_on(node)))

    def visit_IfExpression(self, node: ast.IfExpression):
        # Process a ternary conditional expression (body if test else orelse)
        self.visit(node.test)
        if not self._check_bool(node.test):
            self.report(err.InvalidConditionType(
                self.get_error_info_on(node.test), str(node.test.type_info)
            ))
        self.visit(node.body)
        self.visit(node.orelse)

        body_type = node.body.type_info
        else_type = node.orelse.type_info

        # Both branches must have the same type
        if body_type != else_type:
            node.annotate_type(ErrorTypeInfo())
            return
        
        node.annotate_type(body_type)

    def visit_BinaryExpression(self, node: ast.BinaryExpression):
        # Process binary operations with type checking based on operator type
        # Handles arithmetic, comparison, and boolean operators

        self.visit(node.lhs)
        self.visit(node.rhs)
        
        lhs_type = node.lhs.type_info
        rhs_type = node.rhs.type_info

        if isinstance(lhs_type, ErrorTypeInfo) or isinstance(rhs_type, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        # Check type compatibility for various operator categories
        all_int = isinstance(lhs_type, IntTypeInfo) and isinstance(rhs_type, IntTypeInfo)
        all_bool = isinstance(lhs_type, BoolTypeInfo) and isinstance(rhs_type, BoolTypeInfo)
        all_float = isinstance(lhs_type, FloatTypeInfo) and isinstance(rhs_type, FloatTypeInfo)
        int_float = (
            isinstance(lhs_type, IntTypeInfo) and isinstance(rhs_type, FloatTypeInfo)
            or isinstance(lhs_type, FloatTypeInfo) and isinstance(rhs_type, IntTypeInfo)
        )

        match node.op:
            case ast.ArithmeticOp():
                # Arithmetic: int+int=int, float+float=float, int+float=float
                if all_int:
                    node.annotate_type(IntTypeInfo())
                    return
                if all_float or int_float:
                    node.annotate_type(FloatTypeInfo())
                    return
                
                self.report(err.InvalidBinaryOpTypes(
                    self.get_error_info_on(node),
                    str(lhs_type), str(rhs_type), node.op.token.value
                ))
                node.annotate_type(ErrorTypeInfo())
                return

            case ast.CompareOp():
                # Comparison operators return boolean
                if all_int or all_float or int_float:
                    node.annotate_type(BoolTypeInfo())
                    return
                
                self.report(err.InvalidBinaryOpTypes(
                    self.get_error_info_on(node),
                    str(lhs_type), str(rhs_type), node.op.token.value
                ))
                node.annotate_type(ErrorTypeInfo())
                return
            
            case ast.BinaryBoolOp():
                # Boolean operators (and, or) return boolean
                if all_int or all_bool:
                    node.annotate_type(BoolTypeInfo())
                    return
                
                self.report(err.InvalidBinaryOpTypes(
                    self.get_error_info_on(node),
                    str(lhs_type), str(rhs_type), node.op.token.value
                ))
                node.annotate_type(ErrorTypeInfo())
                return
            
            case _:
                assert False
    
    def visit_UnaryExpression(self, node: ast.UnaryExpression):
        # Process unary operations (+, -, !) with type checking
        self.visit(node.operand)
        op = node.op
        type_info = node.operand.type_info

        if isinstance(type_info, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        match op:
            case ast.PosOp() | ast.NegOp():
                # Unary plus/minus preserve numeric types
                if isinstance(type_info, (IntTypeInfo, FloatTypeInfo)):
                    node.annotate_type(type_info)
                    return
            case ast.NotOp():
                # Logical NOT returns boolean
                node.annotate_type(BoolTypeInfo())
                return
            case _:
                raise NotImplementedError()
            
        self.report(err.InvalidUnaryOpType(
            self.get_error_info_on(node),
            str(type_info), op.token.value
        ))
        node.annotate_type(ErrorTypeInfo())

    def _bind_arguments(
        self,
        name: str,
        params: list[ParameterInfo],
        psargs: list[ast.PositionalArgument],
        kwargs: list[ast.KeywordArgument],
        error_info: err.ErrorInfo,
    ) -> dict[str, ast.Expression]:
        """
        Bind function call arguments to parameters.
        Handles positional arguments, keyword arguments, and default values.
        Reports errors for excess arguments, missing arguments, duplicates, etc.
        """
        
        bound: dict[str, ast.Expression] = {}
        bound_names: set[str] = set()
        nodefaults = len([p for p in params if p.default_node is None])

        # Process positional arguments
        for i, arg in enumerate(psargs):
            if i >= len(params):
                # Too many positional arguments
                if nodefaults == len(params):
                    self.report(err.ExcessPosArgs(error_info, name, len(params), len(psargs)))
                else:
                    self.report(err.ExcessPosArgsDefault(error_info, name, nodefaults, len(params), len(psargs)))
                break
            
            param = params[i]
            param_name = param.name
            
            if param_name in bound_names:
                self.report(err.DuplicateArgument(error_info, param_name))
            
            bound[param_name] = arg.value
            bound_names.add(param_name)
            
        # Process keyword arguments
        unknown_kwargs = []

        for kwarg in kwargs:
            kw_name = kwarg.name
            
            if kw_name in bound_names:
                self.report(err.DuplicateArgument(error_info, kw_name))
                continue
            
            target_param = None
            for param in params:
                if param.name == kw_name:
                    target_param = param
                    break
            
            if target_param is None:
                unknown_kwargs.append(kw_name)
                continue
            
            bound[kw_name] = kwarg.value
            bound_names.add(kw_name)

        # Apply default values for missing parameters
        for param in params:
            param_name = param.name
            if param_name not in bound_names and param.default_node is not None:
                bound[param_name] = param.default_node
                bound_names.add(param_name)

        # Check for missing required parameters
        missing_params = []
        for param in params:
            param_name = param.name
            if param_name not in bound_names and param.default_node is None:
                missing_params.append(param_name)

        if missing_params:
            self.report(err.MissingParams(error_info, name, len(missing_params), missing_params))

        if unknown_kwargs:
            self.report(err.UnexpectedKwarg(error_info, name, unknown_kwargs))

        provided_args = [arg.value for arg in psargs + kwargs]
        # Check for types
        for name, expr in bound.items():
            if expr in provided_args:
                # Only check types for explicitly provided arguments, not defaults
                param_info = next(p for p in params if p.name == name)
                self.visit(expr)
                value_type = expr.type_info
                if value_type != param_info.type_info:
                    self.report(err.UnassignableType(error_info, str(value_type), str(param_info.type_info)))

        return bound
    
    def visit_CallExpression(self, node: ast.CallExpression):
        callee = node.callee

        if not isinstance(callee, ast.Identifier):
            raise NotImplementedError()
        
        name = callee.name
        signature = self.symtab.lookup_func_signature(name)
        if signature is None:
            self.report(err.FunctionNotDeclared(self.get_error_info_on(callee), name))
            node.annotate_type(ErrorTypeInfo())
            return
        
        params = signature.param_infos
        returntype = signature.return_type
        args = node.args

        # Separate positional and keyword arguments
        psargs = [arg for arg in args if isinstance(arg, ast.PositionalArgument)]
        kwargs = [arg for arg in args if isinstance(arg, ast.KeywordArgument)]
        error_info = self.get_error_info_on(node)

        # Bind arguments to parameters and check for errors
        bound = self._bind_arguments(name, params, psargs, kwargs, error_info)
        self.call_param_bindings[node] = bound
        node.annotate_type(returntype)
    
    def visit_Constant(self, node: ast.Constant):
        # Determine the type of a constant literal
        if isinstance(node.type, ast.IntType):
            node.annotate_type(IntTypeInfo())
        elif isinstance(node.type, ast.FloatType):
            node.annotate_type(FloatTypeInfo())
        elif isinstance(node.type, ast.BoolType):
            # Booleans as integers (1 for true, 0 for false)
            node.annotate_type(BoolTypeInfo())
        else:
            assert False

    def visit_Identifier(self, node: ast.Identifier):
        # Resolve an identifier to its declared type
        name = node.name
        type_info = self.lookup_vartype(name)
        if type_info is None:
            self.report(err.VariableNotDeclared(
                self.get_error_info_on(node), name
            ))
            node.annotate_type(ErrorTypeInfo())
            return
        node.annotate_type(type_info)
    
    def visit_IntType(self, node: ast.IntType):
        return IntTypeInfo()
    
    def visit_FloatType(self, node: ast.FloatType):
        return FloatTypeInfo()
    
    def visit_BoolType(self, node: ast.BoolType):
        return BoolTypeInfo()


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
