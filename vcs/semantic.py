from __future__ import annotations

from enum import Enum, auto
from typing import Callable, overload, cast

from vcs.astnodes import *
from vcs.errors import *
from vcs.lexer import TokenType


class TypeInfo:
    """
    Base class for all type information in the semantic analysis phase.
    Represents the type of a variable, expression, or function return value.
    """

class VoidTypeInfo(TypeInfo):
    """
    Represents the void type, used for functions that return no value.
    """
    
    def __str__(self):
        return "Void"
    
    def __eq__(self, other):
        return isinstance(other, VoidTypeInfo)


class IntTypeInfo(TypeInfo):
    """
    Represents the integer type (e.g., 42, -17, 0).
    """
    
    def __str__(self):
        return "Int"
    
    def __eq__(self, other):
        return isinstance(other, IntTypeInfo)


class BoolTypeInfo(TypeInfo):
    """
    Represents the boolean type (true/false).
    """
    
    def __str__(self):
        return "Bool"
    
    def __eq__(self, other):
        return isinstance(other, BoolTypeInfo)


class FloatTypeInfo(TypeInfo):
    """
    Represents the floating-point type (e.g., 3.14, -2.5).
    """
    
    def __str__(self):
        return "Float"
    
    def __eq__(self, other):
        return isinstance(other, FloatTypeInfo)


class ErrorTypeInfo(TypeInfo):
    """
    Represents an error type, used when type checking fails.
    Propagates through the AST to avoid cascading errors.
    """


class ParameterInfo:
    """
    Stores information about a function parameter including its name,
    type, and optional default value expression.
    """
    
    def __init__(self, name: str, type_info: TypeInfo, default_node: Expression | None = None):
        self.name = name
        self.type_info = type_info
        self.default_node = default_node  # Expression for default value, if any


class FunctionTypeInfo(TypeInfo):
    """
    Represents the type signature of a function including parameters and return type.
    """
    
    def __init__(self, param_infos: list[ParameterInfo], return_type: TypeInfo):
        self.param_infos = param_infos
        self.return_type = return_type


class SymbolKind(Enum):
    """
    Enumeration of the different kinds of symbols that can be stored.
    """
    VARIABLE = auto()  # Local or global variable
    PARAMETER = auto()  # Function parameter
    FUNCTION = auto()  # Function declaration


class Symbol:
    """
    Represents a named symbol in the symbol table with its kind and type.
    """
    
    def __init__(self, name: str, kind: SymbolKind, type_info: TypeInfo):
        self.name = name
        self.kind = kind
        self.type_info = type_info


class Scope:
    """
    Represents a lexical scope (e.g., function body, block statement).
    Scopes form a hierarchy through the enclosing reference.
    """
    
    def __init__(self, enclosing: Scope | None = None):
        self.enclosing = enclosing  # Parent scope, or None for global scope
        self._symbols: dict[str, Symbol] = {}  # Symbols declared in this scope

    def declare(self, symbol: Symbol) -> None:
        """Declare a symbol in the current scope"""
        self._symbols[symbol.name] = symbol

    def lookup_local(self, name: str) -> Symbol | None:
        """Look up a symbol only in the current scope (not enclosing scopes)"""
        return self._symbols.get(name)

    def lookup(self, name: str) -> Symbol | None:
        """Look up a symbol in the current scope and all enclosing scopes"""
        scope = self
        while scope is not None:
            sym = scope.lookup_local(name)
            if sym is not None:
                return sym
            scope = scope.enclosing
        return None


class SymbolTable:
    """
    Manages scopes and symbol lookup for the entire program.
    Tracks the current scope and provides methods for entering/exiting scopes.
    """
    
    def __init__(self):
        self.global_scope = Scope(enclosing=None)
        self.cur_scope = self.global_scope

    def enter_scope(self) -> Scope:
        """Create and enter a new nested scope"""
        new_scope = Scope(enclosing=self.cur_scope)
        self.cur_scope = new_scope
        return new_scope

    def exit_scope(self) -> None:
        """Exit the current scope, returning to the enclosing scope"""
        if self.cur_scope.enclosing is not None:
            self.cur_scope = self.cur_scope.enclosing
        else:
            raise RuntimeError("Cannot exit global scope")
    
    def set_scope(self, scope: Scope):
        """Manually set the current scope (used for restoring scope after processing)"""
        self.cur_scope = scope

    def declare_var(self, name: str, type_info: TypeInfo) -> None:
        """Declare a variable in the current scope"""
        sym = Symbol(name, SymbolKind.VARIABLE, type_info)
        self.cur_scope.declare(sym)

    def declare_param(self, name: str, type_info: TypeInfo) -> None:
        """Declare a function parameter in the current scope"""
        sym = Symbol(name, SymbolKind.PARAMETER, type_info)
        self.cur_scope.declare(sym)

    def declare_func(self, name: str, signature: FunctionTypeInfo) -> None:
        """Declare a function in the current scope"""
        sym = Symbol(name, SymbolKind.FUNCTION, signature)
        self.cur_scope.declare(sym)

    def lookup(self, name: str) -> Symbol | None:
        """Look up a symbol in the current scope and enclosing scopes"""
        return self.cur_scope.lookup(name)

    def lookup_func_signature(self, name: str) -> FunctionTypeInfo | None:
        """Look up a function signature by name"""
        sym = self.lookup(name)
        if sym and sym.kind == SymbolKind.FUNCTION:
            return cast(FunctionTypeInfo, sym.type_info)
        return None

    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if a name is declared in the current scope (not enclosing)"""
        return self.cur_scope.lookup_local(name) is not None

    def register_name(self, name: str, type_info: TypeInfo):
        """Register a variable in the current scope (wrapper for declare_var)"""
        self.declare_var(name, type_info)


class SemanticAnalyzer(ASTNodeVisitor):
    """
    Performs semantic analysis and type checking on the AST.
    Verifies type correctness, symbol resolution, and semantic rules.
    """
    
    def __init__(
        self, errors: ErrorCollector,
        get_error_info_on: Callable[[ASTNode | TokenInfo | None], ErrorInfo],
    ):
        self.errors = errors
        self.get_error_info_on = get_error_info_on  # Factory for error location info

        self.symtab = SymbolTable()  # Symbol table for scope management

        self.current_return_type: TypeInfo  # Type expected for return statements
        self.inside_loop = False  # Track if we're inside a loop (for break/continue)

        # Map call expressions to their argument bindings for later analysis
        self.call_param_bindings: dict[CallExpression, dict[str, Expression]] = {}

    def report(self, error: CompilerError):
        """Report a compilation error to the error collector"""
        self.errors.add(error)

    @overload
    def visit(self, node: Module) -> None: ...
    @overload
    def visit(self, node: FunctionDeclaration) -> None: ...
    @overload
    def visit(self, node: Comment) -> None: ...
    @overload
    def visit(self, node: Statement) -> None: ...
    @overload
    def visit(self, node: Expression) -> None: ...
    @overload
    def visit(self, node: Type) -> TypeInfo: ...

    def visit(self, node):
        return super().visit(node)

    def enter_scope(self):
        """Enter a new lexical scope"""
        return self.symtab.enter_scope()
    
    def exit_scope(self):
        """Exit the current lexical scope"""
        self.symtab.exit_scope()
    
    def register_name(self, name: str, type: TypeInfo):
        """Register a variable in the symbol table"""
        self.symtab.declare_var(name, type)

    def declare_func(self, name: str, params: list[Parameter], return_type: TypeInfo):
        """Declare a function in the symbol table with its parameter info"""
        param_infos = []
        for param in params:
            declared_type = self.visit(param.type)
            param_infos.append(ParameterInfo(
                param.name_token.value, declared_type, param.default
            ))
        
        signature = FunctionTypeInfo(param_infos, return_type)
        self.symtab.declare_func(name, signature)

    def declare_var(self, name: str, declared_type: TypeInfo, value: Expression | None):
        """Declare a variable and optionally check its initializer type"""
        self.register_name(name, declared_type)
        
        if isinstance(declared_type, ErrorTypeInfo):
            return  # Don't check if type is already erroneous
        
        if value is not None:
            self.visit(value)
            if isinstance(value.type_info, ErrorTypeInfo):
                return
            
            # Check that initializer type matches declared type
            if value.type_info != declared_type:
                self.report(UnassignableType(
                    self.get_error_info_on(value),
                    str(value.type_info), str(declared_type)
                ))

    def lookup_vartype(self, name: str) -> TypeInfo | None:
        """Look up the type of a variable by name"""
        sym = self.symtab.lookup(name)
        if sym is not None:
            return sym.type_info
        return None

    def check_assignment(self, target: LeftExpression, value: Expression) -> bool:
        # Check that a value can be assigned to a target (type compatibility)
        self.visit(target)
        self.visit(value)

        if target.type_info == value.type_info:
            return True

        if isinstance(target.type_info, ErrorTypeInfo) or isinstance(value.type_info, ErrorTypeInfo):
            return True

        if isinstance(value.type_info, IntTypeInfo) and isinstance(target.type_info, FloatTypeInfo):
            return True

        self.report(UnassignableType(
            self.get_error_info_on(value),
            str(value.type_info), str(target.type_info)
        ))
        return False


    def visit_Module(self, node: Module):
        # Entry point
        scope = self.enter_scope()
        node.annotate_scope(scope)

        # First pass: collect all function signatures
        for sub in node.body:
            if isinstance(sub, FunctionDeclaration):
                return_type = (
                    self.visit(sub.return_type)
                    if sub.return_type is not None
                    else VoidTypeInfo()
                )
                name = sub.name_token.value
                if self.symtab.lookup_func_signature(name) is not None:
                    self.report(FunctionDeclared(
                        self.get_error_info_on(sub.name_token), name
                    ))
                self.declare_func(name, sub.params, return_type)

        # Second pass: process function bodies
        for sub in node.body:
            self.visit(sub)
    
        self.exit_scope()

    def visit_FunctionDeclaration(self, node: FunctionDeclaration):
        # Create scope, declare parameters, and type-check the function body
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
        
        # Type-check the function body if it exists
        if node.body is not None:
            self.visit(node.body)

        self.exit_scope()
    
    def visit_Comment(self, node: Comment):
        pass

    def visit_Block(self, node: Block):
        # Process a block statement as a new lexical scope
        scope = self.enter_scope()
        node.annotate_scope(scope)
        for stmt in node.body:
            self.visit(stmt)
        self.exit_scope()
    
    def visit_ExpressionStatement(self, node: ExpressionStatement):
        # Process an expression statement (expression used as a statement)
        self.visit(node.expression)

    def visit_VariableDeclarationStatement(self, node: VariableDeclarationStatement):
        # Process a variable declaration, checking for redeclaration and type compatibility
        name = node.name_token.value
        if self.symtab.is_declared_in_current_scope(name):
            self.report(VariableDeclared(
                self.get_error_info_on(node.name_token), name
            ))
        declared_type = self.visit(node.type)
        self.declare_var(name, declared_type, node.value)

    def visit_AssignStatement(self, node: AssignStatement):
        # Process a regular assignment statement
        self.check_assignment(node.target, node.value)
    
    def visit_AugAssignStatement(self, node: AugAssignStatement):
        # Process an augmented assignment (e.g., +=, -=, etc.)
        if self.check_assignment(node.target, node.value):
            left_type = node.target.type_info
            right_type = node.value.type_info

            left_int = isinstance(left_type, IntTypeInfo)
            left_float = isinstance(left_type, FloatTypeInfo)
            right_int = isinstance(right_type, IntTypeInfo)
            right_float = isinstance(right_type, FloatTypeInfo)

            if left_int and right_int:
                return
            
            if left_float and right_float:
                return
            
            if left_float and right_int:
                return
            
            self.report(InvalidAugmentedOpTypes(
                self.get_error_info_on(node),
                str(left_type), str(right_type), node.op.token.value
            ))
    
    def visit_PassStatement(self, node: PassStatement):
        pass

    def visit_ReturnStatement(self, node: ReturnStatement):
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
            self.report(UnassignableType(
                self.get_error_info_on(err_node),
                str(value_type), str(self.current_return_type)
            ))
    
    def visit_IfStatement(self, node: IfStatement):
        self.visit(node.test)  # Condition expression
        self.visit(node.body)  # Then branch
        if node.orelse:
            self.visit(node.orelse)  # Else branch (if present)
    
    def visit_WhileStatement(self, node: WhileStatement):
        # Process a while loop, tracking loop context for break/continue
        self.visit(node.test)
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)
        self.inside_loop = was_loop
    
    def visit_ForStatement(self, node: ForStatement):
        # Process a for loop, which has its own scope for initialization
        scope = self.enter_scope()
        node.annotate_scope(scope)
        self.visit(node.init_stmt)  # Initialization statement
        self.visit(node.test)  # Loop condition
        self.visit(node.end_stmt)  # End-of-iteration statement
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)  # Loop body
        self.inside_loop = was_loop
        self.exit_scope()

    def visit_BreakStatement(self, node: BreakStatement):
        # Check that break is used only inside a loop
        if not self.inside_loop:
            self.report(BreakOutsideLoop(self.get_error_info_on(node)))
    
    def visit_ContinueStatement(self, node: ContinueStatement):
        # Check that continue is used only inside a loop
        if not self.inside_loop:
            self.report(ContinueOutsideLoop(self.get_error_info_on(node)))

    def visit_IfExpression(self, node: IfExpression):
        # Process a ternary conditional expression (body if test else orelse)
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

        body_type = node.body.type_info
        else_type = node.orelse.type_info

        # Both branches must have the same type
        if body_type != else_type:
            node.annotate_type(ErrorTypeInfo())
            return
        
        node.annotate_type(body_type)

    def visit_BinaryExpression(self, node: BinaryExpression):
        # Process binary operations with type checking based on operator type
        # Handles arithmetic, comparison, and boolean operators

        self.visit(node.left)
        self.visit(node.right)
        
        left_type = node.left.type_info
        right_type = node.right.type_info

        if isinstance(left_type, ErrorTypeInfo) or isinstance(right_type, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        # Check type compatibility for various operator categories
        all_int = isinstance(left_type, IntTypeInfo) and isinstance(right_type, IntTypeInfo)
        all_float = isinstance(left_type, FloatTypeInfo) and isinstance(right_type, FloatTypeInfo)
        int_float = (
            isinstance(left_type, IntTypeInfo) and isinstance(right_type, FloatTypeInfo)
            or isinstance(left_type, FloatTypeInfo) and isinstance(right_type, IntTypeInfo)
        )

        match node.op:
            case ArithmeticOp():
                # Arithmetic: int+int=int, float+float=float, int+float=float
                if all_int:
                    node.annotate_type(IntTypeInfo())
                    return
                if all_float or int_float:
                    node.annotate_type(FloatTypeInfo())
                    return
                
                self.report(InvalidBinaryOpTypes(
                    self.get_error_info_on(node),
                    str(left_type), str(right_type), node.op.token.value
                ))
                node.annotate_type(ErrorTypeInfo())
                return

            case CompareOp():
                # Comparison operators return boolean
                if all_int or all_float or int_float:
                    node.annotate_type(BoolTypeInfo())
                    return
                
                self.report(InvalidBinaryOpTypes(
                    self.get_error_info_on(node),
                    str(left_type), str(right_type), node.op.token.value
                ))
                node.annotate_type(ErrorTypeInfo())
                return
            
            case BinaryBoolOp():
                # Boolean operators (and, or) return boolean
                node.annotate_type(BoolTypeInfo())
    
    def visit_UnaryExpression(self, node: UnaryExpression):
        # Process unary operations (+, -, !) with type checking
        self.visit(node.operand)
        op = node.op
        type_info = node.operand.type_info

        if isinstance(type_info, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        match op:
            case PosOp() | NegOp():
                # Unary plus/minus preserve numeric types
                if isinstance(type_info, (IntTypeInfo, FloatTypeInfo)):
                    node.annotate_type(type_info)
                    return
            case NotOp():
                # Logical NOT returns boolean
                node.annotate_type(BoolTypeInfo())
                return
            case _:
                raise NotImplementedError()
            
        self.report(InvalidUnaryOpType(
            self.get_error_info_on(node),
            str(type_info), op.token.value
        ))
        node.annotate_type(ErrorTypeInfo())

    def _bind_arguments(
        self,
        name: str,
        params: list[ParameterInfo],
        psargs: list[PositionalArgument],
        kwargs: list[KeywordArgument],
        error_info: ErrorInfo,
    ) -> dict[str, Expression]:
        """
        Bind function call arguments to parameters.
        Handles positional arguments, keyword arguments, and default values.
        Reports errors for excess arguments, missing arguments, duplicates, etc.
        """
        
        bound: dict[str, Expression] = {}
        bound_names: set[str] = set()
        nodefaults = len([p for p in params if p.default_node is None])

        # Process positional arguments
        for i, arg in enumerate(psargs):
            if i >= len(params):
                # Too many positional arguments
                if nodefaults == len(params):
                    self.report(ExcessPosArgs(error_info, name, len(params), len(psargs)))
                else:
                    self.report(ExcessPosArgsDefault(error_info, name, nodefaults, len(params), len(psargs)))
                break
            
            param = params[i]
            param_name = param.name
            
            if param_name in bound_names:
                self.report(DuplicateArgument(error_info, param_name))
            
            bound[param_name] = arg.value
            bound_names.add(param_name)
            
        # Process keyword arguments
        unknown_kwargs = []

        for kwarg in kwargs:
            kw_name = kwarg.name_token.value
            
            if kw_name in bound_names:
                self.report(DuplicateArgument(error_info, kw_name))
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
            self.report(MissingParams(error_info, name, len(missing_params), missing_params))

        if unknown_kwargs:
            self.report(UnexpectedKwarg(error_info, name, unknown_kwargs))

        # Check for types
        for name, expr in bound.items():
            if expr in [arg.value for arg in psargs + kwargs]:
                param_info = next(p for p in params if p.name == name)
                self.visit(expr)
                value_type = expr.type_info
                if value_type != param_info.type_info:
                    self.report(UnassignableType(
                        error_info, str(value_type), str(param_info.type_info
                    )))

        return bound
    
    def visit_CallExpression(self, node: CallExpression):
        callee = node.callee

        if not isinstance(callee, Identifier):
            raise NotImplementedError()
        
        name = callee.token.value
        signature = self.symtab.lookup_func_signature(name)
        if signature is None:
            self.report(FunctionNotDeclared(self.get_error_info_on(callee), name))
            node.annotate_type(ErrorTypeInfo())
            return
        
        params = signature.param_infos
        returntype = signature.return_type
        args = node.args

        # Separate positional and keyword arguments
        psargs = [arg for arg in args if isinstance(arg, PositionalArgument)]
        kwargs = [arg for arg in args if isinstance(arg, KeywordArgument)]
        error_info = self.get_error_info_on(node)

        # Bind arguments to parameters and check for errors
        bound = self._bind_arguments(name, params, psargs, kwargs, error_info)
        self.call_param_bindings[node] = bound
        node.annotate_type(returntype)
    
    def visit_Constant(self, node: Constant):
        # Determine the type of a constant literal
        value = node.value.value

        if node.value.type == TokenType.INT:
            node.annotate_type(IntTypeInfo())
        elif node.value.type == TokenType.FLOAT:
            node.annotate_type(FloatTypeInfo())
        elif value in ["true", "false"]:
            node.annotate_type(BoolTypeInfo())
        else:
            raise NotImplementedError()
    
    def visit_Identifier(self, node: Identifier):
        # Resolve an identifier to its declared type
        name = node.token.value
        type_info = self.lookup_vartype(name)
        if type_info is None:
            self.report(VariableNotDeclared(
                self.get_error_info_on(node), name
            ))
            node.annotate_type(ErrorTypeInfo())
            return
        node.annotate_type(type_info)
    
    def visit_IntType(self, node: IntType):
        return IntTypeInfo()
    
    def visit_FloatType(self, node: FloatType):
        return FloatTypeInfo()
    
    def visit_BoolType(self, node: BoolType):
        return BoolTypeInfo()


def main():
    import argparse
    import sys
    
    from vcs.lexer import Lexer
    from vcs.parser import Parser
    
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(dest="filename", metavar="filename.vcs")
    argparser.add_argument(
        "-s",
        "--skip-comments",
        action="store_true",
        help="Skip all the comment tokens",
    )
    argparser.add_argument(
        "-t",
        "--tabsize",
        metavar="TABSIZE",
        default=4,
        help="How many spaces per tab character",
    )
    args = argparser.parse_args()

    filename = args.filename
    skip_comments = args.skip_comments

    # Read source file
    with open(filename, encoding="utf8") as f:
        source = f.read()

    # Lex, parse, and perform semantic analysis
    errors = ErrorCollector()
    lexer = Lexer(source, filename, errors, args.tabsize)
    parser = Parser(lexer, errors, skip_comments)
    tree: Module = parser.parse()
    typechecker = SemanticAnalyzer(errors, parser.get_error_info_on)
    typechecker.visit(tree)

    # Report any errors found during analysis
    if not errors.ok():
        errors.sort()
        for issue in errors.issues:
            print(dump_error(issue), file=sys.stderr)
        return

if __name__ == "__main__":
    main()
