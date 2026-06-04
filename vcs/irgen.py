from __future__ import annotations
from contextlib import contextmanager

from vcs import astnodes as ast
from vcs import ir
from vcs import semantic as sem
from vcs.errors import *
from vcs.lexer import TokenType


class IRGenerator(ast.ASTNodeVisitor):
    def __init__(
        self,
        namespace: str,
        symtab: sem.SymbolTable,
        call_param_bindings: dict[ast.CallExpression, dict[str, ast.Expression]],
    ):
        self.namespace = namespace
        self.symtab = symtab
        self.call_param_bindings = call_param_bindings

        self.module = ir.Module(namespace)
        self.builder = ir.IRBuilder()

        self.cur_func: ir.Function

        self._symbol_map: dict[sem.Symbol, ir.NamedValue] = {}
        self._func_map: dict[str, ir.Function] = {}
        self._loop_stack: list[tuple[ir.BasicBlock, ir.BasicBlock]] = []

        self._block_id = 0

    def emit(self, inst: ir.Instruction):
        self.builder.emit(inst)

    def _is_terminated(self) -> bool:
        if not self.builder.current_block.instructions:
            return False
        last = self.builder.current_block.instructions[-1]
        return isinstance(last, (ir.Return, ir.Branch, ir.Goto))


    def _get_irtype(self, semtype: sem.TypeInfo) -> ir.Type:
        if isinstance(semtype, (sem.IntTypeInfo, sem.BoolTypeInfo)):
            return ir.IntType()
        if isinstance(semtype, sem.FloatTypeInfo):
            return ir.FloatType()
        if isinstance(semtype, sem.VoidTypeInfo):
            return ir.VoidType()
        raise ValueError(f"Unsupported semantic type: {semtype}")

    def _get_irfunc(self, name: str, signature: sem.FunctionTypeInfo) -> ir.Function:
        return_type = self._get_irtype(signature.return_type)
        param_types = [self._get_irtype(p.type_info) for p in signature.param_infos]
        param_names = [p.name for p in signature.param_infos]
        func_type = ir.FunctionType(return_type, param_types)
        return ir.Function(name, func_type, param_names)

    def _new_block(self) -> ir.BasicBlock:
        name = f"L{self._block_id}"
        self._block_id += 1
        return self.cur_func.create_block(name)

    @contextmanager
    def _with_scope(self, scope: sem.Scope):
        prev = self.symtab.cur_scope
        self.symtab.set_scope(scope)
        yield scope
        self.symtab.set_scope(prev)

    def visit_Module(self, node: ast.Module):
        for sub in node.body:
            if isinstance(sub, ast.FunctionDeclaration):
                func = self._get_irfunc(sub.name_token.value, sub.signature)
                self.module.add_function(func)
                self._func_map[sub.name_token.value] = func

        for sub in node.body:
            self.visit(sub)

    def visit_FunctionDeclaration(self, node: ast.FunctionDeclaration):
        func = self._func_map[node.name_token.value]
        self.cur_func = func
        self.builder.set_block(func.entry_block)

        with self._with_scope(node.scope):
            for param_info in node.signature.param_infos:
                symbol = self.symtab.lookup(param_info.name)
                assert symbol is not None, f"Parameter {param_info.name} not found"
                irtype = self._get_irtype(param_info.type_info)
                param_val = ir.NamedValue(param_info.name, irtype)
                self._symbol_map[symbol] = param_val

            if node.body is not None:
                self.visit(node.body)

        if not self._is_terminated():
            if isinstance(node.signature.return_type, sem.VoidTypeInfo):
                self.emit(ir.Return())
            else:
                pass

    def visit_Comment(self, node: ast.Comment):
        self.emit(ir.Comment(node.token.value))

    def visit_Block(self, node: ast.Block):
        with self._with_scope(node.scope):
            for stmt in node.body:
                self.visit(stmt)

    def visit_ExpressionStatement(self, node: ast.ExpressionStatement):
        self.visit(node.expression)

    def visit_VariableDeclarationStatement(self, node: ast.VariableDeclarationStatement):
        name = node.name_token.value
        symbol = self.symtab.lookup(name)
        assert symbol is not None
        irtype = self._get_irtype(symbol.type_info)

        target = ir.NamedValue(name, irtype)
        self._symbol_map[symbol] = target

        if node.value is not None:
            value = self.visit(node.value)
            self.emit(ir.Load(value, target))

    def visit_AssignStatement(self, node: ast.AssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self.emit(ir.Load(value, target))

    def visit_AugAssignStatement(self, node: ast.AugAssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)

        target_sem = node.target.type_info
        value_sem = node.value.type_info
        int_int = isinstance(target_sem, sem.IntTypeInfo) and isinstance(value_sem, sem.IntTypeInfo)
        float_float = isinstance(target_sem, sem.FloatTypeInfo) and isinstance(value_sem, sem.FloatTypeInfo)
        float_int = isinstance(target_sem, sem.FloatTypeInfo) and isinstance(value_sem, sem.IntTypeInfo)

        op_map = {
            ast.AddOp: (ir.IAdd, ir.FAdd),
            ast.SubOp: (ir.ISub, ir.FSub),
            ast.MulOp: (ir.IMul, ir.FMul),
            ast.DivOp: (ir.IDiv, ir.FDiv),
            ast.ModOp: (ir.IMod, ir.FMod),
        }
        i_op, f_op = op_map[type(node.op)]

        if int_int:
            self.emit(i_op(target, value, target))
        elif float_float:
            self.emit(f_op(target, value, target))
        elif float_int:
            temp = self.builder.new_temp(ir.FloatType())
            self.emit(ir.IntToFloat(value, temp))
            self.emit(f_op(target, temp, target))
        else:
            raise RuntimeError("Invalid augmented assignment types")

    def visit_PassStatement(self, node: ast.PassStatement):
        pass

    def visit_ReturnStatement(self, node: ast.ReturnStatement):
        if node.value is None:
            self.emit(ir.Return())
        else:
            ret_val = self.visit(node.value)
            self.emit(ir.Return(ret_val))

    def visit_IfStatement(self, node: ast.IfStatement):
        test = self.visit(node.test)
        if not isinstance(test, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self.emit(ir.Load(test, temp))
            test = temp

        merge_block = self._new_block()
        if node.orelse:
            else_block = self._new_block()
        else:
            else_block = merge_block

        then_block = self._new_block()
        self.emit(ir.Branch(test, then_block, else_block))

        self.builder.set_block(then_block)
        self.visit(node.body)
        if not self._is_terminated():
            self.emit(ir.Goto(merge_block))

        if node.orelse:
            self.builder.set_block(else_block)
            self.visit(node.orelse)
            if not self._is_terminated():
                self.emit(ir.Goto(merge_block))

        self.builder.set_block(merge_block)

    def visit_WhileStatement(self, node: ast.WhileStatement):
        cond_block = self._new_block()
        body_block = self._new_block()
        end_block = self._new_block()

        self._loop_stack.append((end_block, cond_block))

        self.emit(ir.Goto(cond_block))
        self.builder.set_block(cond_block)

        test = self.visit(node.test)
        if not isinstance(test, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self.emit(ir.Load(test, temp))
            test = temp
        self.emit(ir.Branch(test, body_block, end_block))

        self.builder.set_block(body_block)
        self.visit(node.body)
        if not self._is_terminated():
            self.emit(ir.Goto(cond_block))

        self._loop_stack.pop()
        self.builder.set_block(end_block)

    def visit_ForStatement(self, node: ast.ForStatement):
        cond_block = self._new_block()
        body_block = self._new_block()
        update_block = self._new_block()
        end_block = self._new_block()

        self._loop_stack.append((end_block, update_block))

        self.visit(node.init_stmt)
        self.emit(ir.Goto(cond_block))
        self.builder.set_block(cond_block)

        test = self.visit(node.test)
        if not isinstance(test, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self.emit(ir.Load(test, temp))
            test = temp
        self.emit(ir.Branch(test, body_block, end_block))

        self.builder.set_block(body_block)
        self.visit(node.body)
        if not self._is_terminated():
            self.emit(ir.Goto(update_block))

        self.builder.set_block(update_block)
        self.visit(node.end_stmt)
        self.emit(ir.Goto(cond_block))

        self._loop_stack.pop()
        self.builder.set_block(end_block)

    def visit_BreakStatement(self, node: ast.BreakStatement):
        if not self._loop_stack:
            return
        self.emit(ir.Goto(self._loop_stack[-1][0]))

    def visit_ContinueStatement(self, node: ast.ContinueStatement):
        if not self._loop_stack:
            return
        self.emit(ir.Goto(self._loop_stack[-1][1]))

    def visit_BinaryExpression(self, node: ast.BinaryExpression) -> ir.Value:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        target_type = self._get_irtype(node.type_info)
        target = self.builder.new_temp(target_type)

        left_sem = node.left.type_info
        right_sem = node.right.type_info

        def is_int(t): return isinstance(t, sem.IntTypeInfo)
        def is_float(t): return isinstance(t, sem.FloatTypeInfo)

        if isinstance(node.op, ast.ArithmeticOp):
            i_op, f_op = {
                ast.AddOp: (ir.IAdd, ir.FAdd),
                ast.SubOp: (ir.ISub, ir.FSub),
                ast.MulOp: (ir.IMul, ir.FMul),
                ast.DivOp: (ir.IDiv, ir.FDiv),
                ast.ModOp: (ir.IMod, ir.FMod),
            }[type(node.op)]

            if is_int(left_sem) and is_int(right_sem):
                self.emit(i_op(lhs, rhs, target))
            elif is_float(left_sem) and is_float(right_sem):
                self.emit(f_op(lhs, rhs, target))
            else:
                if is_int(left_sem):
                    temp = self.builder.new_temp(ir.FloatType())
                    self.emit(ir.IntToFloat(lhs, temp))
                    lhs = temp
                if is_int(right_sem):
                    temp = self.builder.new_temp(ir.FloatType())
                    self.emit(ir.IntToFloat(rhs, temp))
                    rhs = temp
                self.emit(f_op(lhs, rhs, target))

        elif isinstance(node.op, ast.CompareOp):
            op_str = {
                ast.EqOp: 'eq',
                ast.NegOp: 'ne',
                ast.LtOp: 'lt',
                ast.GtOp: 'gt',
                ast.LtEOp: 'le',
                ast.GtEOp: 'ge',
            }[type(node.op)]

            if is_int(left_sem) and is_int(right_sem):
                self.emit(ir.ICmp(lhs, rhs, target, op_str))
            else:
                if is_int(left_sem):
                    temp = self.builder.new_temp(ir.FloatType())
                    self.emit(ir.IntToFloat(lhs, temp))
                    lhs = temp
                if is_int(right_sem):
                    temp = self.builder.new_temp(ir.FloatType())
                    self.emit(ir.IntToFloat(rhs, temp))
                    rhs = temp
                self.emit(ir.FCmp(lhs, rhs, target, op_str))

        elif isinstance(node.op, ast.BinaryBoolOp):
            if isinstance(node.op, ast.AndOp):
                self.emit(ir.And(lhs, rhs, target))
            else:
                self.emit(ir.Or(lhs, rhs, target))
        else:
            raise NotImplementedError(f"Unsupported binary operator: {node.op}")

        return target

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> ir.Value:
        operand = self.visit(node.operand)
        target_type = self._get_irtype(node.type_info)
        target = self.builder.new_temp(target_type)

        if isinstance(node.op, ast.NotOp):
            self.emit(ir.Not(operand, target))
        elif isinstance(node.op, ast.NegOp):
            sem_type = node.operand.type_info
            if isinstance(sem_type, sem.IntTypeInfo):
                zero = ir.Constant(0, ir.IntType())
                self.emit(ir.ISub(zero, operand, target))
            else:
                zero = ir.Constant(0.0, ir.FloatType())
                self.emit(ir.FSub(zero, operand, target))
        elif isinstance(node.op, ast.PosOp):
            self.emit(ir.Load(operand, target))
        else:
            raise NotImplementedError(f"Unary operator {node.op}")

        return target

    def visit_CallExpression(self, node: ast.CallExpression) -> ir.Value | None:
        callee = node.callee
        assert isinstance(callee, ast.Identifier)
        func_name = callee.token.value

        ir_func = self._func_map.get(func_name)
        if ir_func is None:
            raise RuntimeError(f"IR function '{func_name}' not found")

        bound = self.call_param_bindings.get(node)
        assert bound is not None
        signature = self.symtab.lookup_func_signature(func_name)
        assert signature is not None
        args = [self.visit(bound[p.name]) for p in signature.param_infos]

        return_type = ir_func.type.return_type
        if isinstance(return_type, ir.VoidType):
            self.emit(ir.Call(ir_func, args))
            return None
        else:
            target = self.builder.new_temp(return_type)
            self.emit(ir.Call(ir_func, args, target))
            return target

    def visit_Constant(self, node: ast.Constant) -> ir.Constant:
        value = node.value.value
        if node.value.type == TokenType.INT:
            return ir.Constant(int(value), ir.IntType())
        elif node.value.type == TokenType.FLOAT:
            return ir.Constant(float(value), ir.FloatType())
        elif value in ('true', 'false'):
            return ir.Constant(1 if value == 'true' else 0, ir.IntType())
        else:
            raise NotImplementedError(f"Unsupported constant: {value}")

    def visit_Identifier(self, node: ast.Identifier) -> ir.NamedValue:
        name = node.token.value
        symbol = self.symtab.lookup(name)
        assert symbol is not None, f"Identifier '{name}' not in symbol table"
        ir_val = self._symbol_map.get(symbol)
        assert ir_val is not None, f"Identifier '{name}' has no IR mapping"
        return ir_val

    def generic_visit(self, node: ast.ASTNode, *args, **kwargs):
        raise NotImplementedError(f"IR codegen for {type(node).__name__} not implemented")


def main():
    import argparse
    import sys

    from vcs.lexer import Lexer
    from vcs.parser import Parser
    from vcs.semantic import SemanticAnalyzer

    argparser = argparse.ArgumentParser()
    argparser.add_argument(dest="filename", metavar="filename.vcs")
    argparser.add_argument("-s", "--skip-comments", action="store_true")
    argparser.add_argument("-t", "--tabsize", metavar="TABSIZE", default=4)
    argparser.add_argument("-n", "--namespace", metavar="NAMESPACE", required=True)
    args = argparser.parse_args()

    filename = args.filename
    namespace = args.namespace

    with open(filename, encoding="utf8") as f:
        source = f.read()

    errors = ErrorCollector()
    lexer = Lexer(source, filename, errors, args.tabsize)
    parser = Parser(lexer, errors, args.skip_comments)
    tree: ast.Module = parser.parse()

    typechecker = SemanticAnalyzer(errors, parser.get_error_info_on)
    typechecker.visit(tree)
    if not errors.ok():
        for issue in errors.issues:
            print(dump_error(issue), file=sys.stderr)
        return
    
    irgen = IRGenerator(
        namespace,
        typechecker.symtab,
        typechecker.call_param_bindings,
    )
    irgen.visit(tree)

    print(irgen.module)


if __name__ == "__main__":
    main()
