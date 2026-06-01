from typing import Callable, overload

from vcs.astnodes import *
from vcs.errors import *
from vcs.lexer import TokenType


class TypeInfo: ...


class VoidTypeInfo(TypeInfo):
    def __str__(self):
        return "Void"
    
    def __eq__(self, other):
        return isinstance(other, VoidTypeInfo)


class IntTypeInfo(TypeInfo):
    def __str__(self):
        return "Int"
    
    def __eq__(self, other):
        return isinstance(other, IntTypeInfo)


class BoolTypeInfo(TypeInfo):
    def __str__(self):
        return "Bool"
    
    def __eq__(self, other):
        return isinstance(other, BoolTypeInfo)


class FloatTypeInfo(TypeInfo):
    def __str__(self):
        return "Float"
    
    def __eq__(self, other):
        return isinstance(other, FloatTypeInfo)


class ErrorTypeInfo(TypeInfo): ...


class ParameterInfo:
    def __init__(self, name: str, type_info: TypeInfo, default_node: Expression | None = None):
        self.name = name
        self.type_info = type_info
        self.default_node = default_node


class FunctionSignature:
    def __init__(self, param_infos: list[ParameterInfo], return_type: TypeInfo):
        self.param_infos = param_infos
        self.return_type = return_type


class SemanticAnalyzer(ASTNodeVisitor):
    def __init__(
        self, errors: ErrorCollector,
        get_error_info_on: Callable[[ASTNode | TokenInfo | None], ErrorInfo],
    ):
        self.errors = errors
        self.get_error_info_on = get_error_info_on

        self.func_signatures: dict[str, FunctionSignature] = {}
        self.scopes: list[dict[str, TypeInfo]] = []

        self.current_return_type: TypeInfo
        self.inside_loop = False

        self.call_param_bindings: dict[CallExpression, dict[str, Expression]] = {}

    def report(self, error: CompilerError):
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

    def push_scope(self):
        self.scopes.append({})
    
    def pop_scope(self):
        self.scopes.pop()
    
    def register_name(self, name: str, type: TypeInfo):
        self.scopes[-1][name] = type

    def declare_function(self, name: str, params: list[Parameter], return_type: TypeInfo):
        param_infos = []
        for param in params:
            declared_type = self.visit(param.type)
            param_infos.append(ParameterInfo(
                param.name_token.value, declared_type, param.default
            ))
        
        self.func_signatures[name] = FunctionSignature(param_infos, return_type)

    def declare_var(self, name: str, declared_type: TypeInfo, value: Expression | None):
        self.register_name(name, declared_type)
        
        if isinstance(declared_type, ErrorTypeInfo):
            return
        
        if value is not None:
            self.visit(value)
            if isinstance(value.type_info, ErrorTypeInfo):
                return
            
            if value.type_info != declared_type:
                self.report(UnassignableType(
                    self.get_error_info_on(value),
                    str(value.type_info), str(declared_type)
                ))

    def lookup_vartype(self, name: str) -> TypeInfo | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_Module(self, node: Module):
        self.push_scope()

        for sub in node.body:
            if isinstance(sub, FunctionDeclaration):
                return_type = (
                    self.visit(sub.return_type)
                    if sub.return_type is not None
                    else VoidTypeInfo()
                )
                name = sub.name_token.value
                if name in self.func_signatures:
                    self.report(FunctionDeclared(
                        self.get_error_info_on(sub.name_token), name
                    ))
                self.declare_function(name, sub.params, return_type)

        for sub in node.body:
            self.visit(sub)
    
        self.pop_scope()

    def visit_FunctionDeclaration(self, node: FunctionDeclaration):
        self.push_scope()
        name = node.name_token.value
        signature = self.func_signatures[name]
        self.current_return_type = signature.return_type
        for param_info in signature.param_infos:
            self.declare_var(
                param_info.name, param_info.type_info, param_info.default_node
            )
        if node.body is not None:
            self.visit(node.body)
        self.pop_scope()
    
    def visit_Comment(self, node: Comment):
        pass

    def visit_Block(self, node: Block):
        self.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()
    
    def visit_ExpressionStatement(self, node: ExpressionStatement):
        self.visit(node.expression)

    def visit_VariableDeclarationStatement(self, node: VariableDeclarationStatement):
        name = node.name_token.value
        if name in self.scopes[-1]:
            self.report(VariableDeclared(
                self.get_error_info_on(node.name_token), name
            ))
        declared_type = self.visit(node.type)
        self.declare_var(name, declared_type, node.value)

    def check_assignment(self, target: LeftExpression, value: Expression):
        self.visit(target)
        self.visit(value)

        if isinstance(target.type_info, ErrorTypeInfo):
            return
        if isinstance(value.type_info, ErrorTypeInfo):
            return

        if target.type_info != value.type_info:
            self.report(UnassignableType(
                self.get_error_info_on(value),
                str(value.type_info), str(target.type_info)
            ))

    def visit_AssignStatement(self, node: AssignStatement):
        self.check_assignment(node.target, node.value)
    
    def visit_AugAssignStatement(self, node: AugAssignStatement):
        self.check_assignment(node.target, node.value)
    
    def visit_SwapStatement(self, node: SwapStatement):
        self.check_assignment(node.left, node.right)
    
    def visit_PassStatement(self, node: PassStatement):
        pass

    def visit_ReturnStatement(self, node: ReturnStatement):
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
        self.visit(node.test)
        self.visit(node.body)
        if node.orelse:
            self.visit(node.orelse)
    
    def visit_WhileStatement(self, node: WhileStatement):
        self.visit(node.test)
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)
        self.inside_loop = was_loop
    
    def visit_ForStatement(self, node: ForStatement):
        self.push_scope()
        self.visit(node.init_stmt)
        self.visit(node.test)
        self.visit(node.end_stmt)
        was_loop = self.inside_loop
        self.inside_loop = True
        self.visit(node.body)
        self.inside_loop = was_loop
        self.pop_scope()

    def visit_BreakStatement(self, node: BreakStatement):
        if not self.inside_loop:
            self.report(BreakOutsideLoop(self.get_error_info_on(node)))
    
    def visit_ContinueStatement(self, node: ContinueStatement):
        if not self.inside_loop:
            self.report(ContinueOutsideLoop(self.get_error_info_on(node)))

    def visit_IfExpression(self, node: IfExpression):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

        body_type = node.body.type_info
        else_type = node.orelse.type_info

        if body_type != else_type:
            node.annotate_type(ErrorTypeInfo())
            return
        
        node.annotate_type(body_type)

    def visit_BinaryExpression(self, node: BinaryExpression):
        self.visit(node.left)
        self.visit(node.right)
        
        left_type = node.left.type_info
        right_type = node.right.type_info

        if isinstance(left_type, ErrorTypeInfo) or isinstance(right_type, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        all_int = isinstance(left_type, IntTypeInfo) and isinstance(right_type, IntTypeInfo)
        all_float = isinstance(left_type, FloatTypeInfo) and isinstance(right_type, FloatTypeInfo)
        int_float = (
            isinstance(left_type, IntTypeInfo) and isinstance(right_type, FloatTypeInfo)
            or isinstance(left_type, FloatTypeInfo) and isinstance(right_type, IntTypeInfo)
        )

        match node.op:
            case ArithmeticOp():
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
                node.annotate_type(BoolTypeInfo())
    
    def visit_UnaryExpression(self, node: UnaryExpression):
        self.visit(node.operand)
        op = node.op
        type_info = node.operand.type_info

        if isinstance(type_info, ErrorTypeInfo):
            node.annotate_type(ErrorTypeInfo())
            return

        match op:
            case PosOp() | NegOp():
                if isinstance(type_info, (IntTypeInfo, FloatTypeInfo)):
                    node.annotate_type(type_info)
                    return
            case NotOp():
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
        
        bound: dict[str, Expression] = {}
        bound_names: set[str] = set()
        nodefaults = len([p for p in params if p.default_node is None])

        for i, arg in enumerate(psargs):
            if i >= len(params):
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

        for param in params:
            param_name = param.name
            if param_name not in bound_names and param.default_node is not None:
                bound[param_name] = param.default_node
                bound_names.add(param_name)

        missing_params = []
        for param in params:
            param_name = param.name
            if param_name not in bound_names and param.default_node is None:
                missing_params.append(param_name)

        if missing_params:
            self.report(MissingParams(error_info, name, len(missing_params), missing_params))

        if unknown_kwargs:
            self.report(UnexpectedKwarg(error_info, name, unknown_kwargs))

        return bound
    
    def visit_CallExpression(self, node: CallExpression):
        callee = node.callee

        if not isinstance(callee, Identifier):
            raise NotImplementedError()
        
        name = callee.token.value
        if name not in self.func_signatures:
            self.report(FunctionNotDeclared(
                self.get_error_info_on(callee), name
            ))
            node.annotate_type(ErrorTypeInfo())
            return
        
        signature = self.func_signatures[name]
        params = signature.param_infos
        returntype = signature.return_type
        args = node.args

        psargs = [arg for arg in args if isinstance(arg, PositionalArgument)]
        kwargs = [arg for arg in args if isinstance(arg, KeywordArgument)]
        error_info = self.get_error_info_on(node)

        bound = self._bind_arguments(name, params, psargs, kwargs, error_info)
        self.call_param_bindings[node] = bound

        node.annotate_type(returntype)
    
    def visit_Constant(self, node: Constant):
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

    # Lex, parse, and generate IR
    errors = ErrorCollector()
    lexer = Lexer(source, filename, errors, args.tabsize)
    parser = Parser(lexer, errors, skip_comments)
    tree: Module = parser.parse()
    typechecker = SemanticAnalyzer(errors, parser.get_error_info_on)
    typechecker.visit(tree)

    # Report any errors
    if not errors.ok():
        errors.sort()
        for issue in errors.issues:
            print(dump_error(issue), file=sys.stderr)
        return

if __name__ == "__main__":
    main()
