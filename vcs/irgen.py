from __future__ import annotations

from contextlib import contextmanager
from typing import cast, overload

from vcs import ast
from vcs import ir
from vcs import semantic as sem
from vcs import utils


class IRGenerator(ast.ASTNodeVisitor):
    """
    Generates IR from AST.
    """
    
    def __init__(
        self,
        namespace: str,
        typechecker: sem.SemanticAnalyzer
    ):
        self.namespace = namespace
        self.typechecker = typechecker
        self.symtab = typechecker.symtab  # Symbol table for name resolution
        self.call_param_bindings = typechecker.call_param_bindings  # Maps call nodes to parameter bindings

        # IR module and builder
        self.module = ir.Module(namespace)  # Container for all IR functions
        self.builder = ir.IRBuilder()  # Helper for constructing IR instructions
        self.cur_func: ir.Function  # Currently active function being generated

        # Maps between compiler abstractions
        self._symbol_map: dict[sem.Symbol, ir.NamedValue] = {}  # Semantic symbol -> IR value
        self._func_map: dict[str, ir.Function] = {}  # Function name -> IR function
        self._loop_stack: list[tuple[ir.BasicBlock, ir.BasicBlock]] = []  # Stack for break/continue: (break_block, continue_block)
        self._block_id = 0  # Counter for generating unique basic block names

    def generate(self) -> ir.Module | None:
        """
        Run the semantic analysis and generate IR.
        """
        tree = self.typechecker.analyze()
        if tree is None:
            return None
        self.visit(tree)
        return self.module

    @overload
    def visit(self, node: ast.Module) -> None: ...
    @overload
    def visit(self, node: ast.FunctionDeclaration) -> None: ...
    @overload
    def visit(self, node: ast.Comment) -> None: ...
    @overload
    def visit(self, node: ast.Statement) -> None: ...
    @overload
    def visit(self, node: ast.LeftExpression) -> ir.NamedValue: ...
    @overload
    def visit(self, node: ast.Expression) -> ir.Value: ...

    def visit(self, node):
        return super().visit(node)

    def emit(self, inst: ir.Instruction):
        self.builder.emit(inst)

    def is_terminated(self) -> bool:
        return self.builder.is_terminated()

    @overload
    def _get_irtype(self, semtype: sem.IntTypeInfo | sem.BoolTypeInfo) -> ir.IntType: ...
    @overload
    def _get_irtype(self, semtype: sem.FloatTypeInfo) -> ir.FloatType: ...
    @overload
    def _get_irtype(self, semtype: sem.VoidTypeInfo) -> ir.VoidType: ...
    @overload
    def _get_irtype(self, semtype: sem.TypeInfo) -> ir.Type: ...

    def _get_irtype(self, semtype: sem.TypeInfo) -> ir.Type:
        if isinstance(semtype, (sem.IntTypeInfo, sem.BoolTypeInfo)):
            return ir.IntType()  # Booleans are represented as integers (0/1)
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

    def new_block(self) -> ir.BasicBlock:
        name = f"l{self._block_id}"
        self._block_id += 1
        return self.cur_func.create_block(name)

    def with_scope(self, scope: sem.Scope):
        return self.symtab.with_scope(scope)

    def visit_Module(self, node: ast.Module):
        # First pass: create function declarations
        for sub in node.body:
            if isinstance(sub, ast.FunctionDeclaration):
                func = self._get_irfunc(sub.name, sub.signature)
                self.module.add_func(func)
                self._func_map[sub.name] = func
                
                # Map function parameters to IR values
                with self.with_scope(sub.scope):
                    for param_info in sub.signature.param_infos:
                        symbol = self.symtab.lookup(param_info.name)
                        assert symbol is not None, f"Parameter {param_info.name} not found"
                        irtype = self._get_irtype(param_info.type_info)
                        param_val = ir.NamedValue(param_info.name, irtype)
                        self._symbol_map[symbol] = param_val

        # Second pass: generate function bodies
        for sub in node.body:
            self.visit(sub)

    def visit_FunctionDeclaration(self, node: ast.FunctionDeclaration):
        func = self._func_map[node.name]
        self.cur_func = func
        self.builder.set_block(func.entry_block)

        with self.with_scope(node.scope):
            # Generate the function body if it exists
            if node.body is not None:
                self.visit(node.body)

        # Add an implicit return if the function doesn't end with a terminator
        if not self.is_terminated():
            if isinstance(node.signature.return_type, sem.VoidTypeInfo):
                self.emit(ir.Return())
            # Non-void functions without a return will have an error elsewhere

    def visit_Comment(self, node: ast.Comment):
        self.emit(ir.Comment(node.value))

    def visit_Block(self, node: ast.Block):
        with self.with_scope(node.scope):
            for stmt in node.body:
                self.visit(stmt)

    def visit_ExpressionStatement(self, node: ast.ExpressionStatement):
        self.visit(node.expression)

    def _emit_assign[T: ir.Type](self, value: ir.Value[T], target: ir.NamedValue[T]):
        if ir.int_typed(value) and ir.int_typed(target):
            self.emit(ir.IntAssign(value, target))
        elif ir.float_typed(value) and ir.float_typed(target):
            self.emit(ir.FloatAssign(value, target))
        else:
            assert False

    def visit_VariableDeclarationStatement(self, node: ast.VariableDeclarationStatement):
        name = node.name
        symbol = self.symtab.lookup(name)
        assert symbol is not None
        irtype = self._get_irtype(symbol.type_info)

        # Create the variable
        target = ir.NamedValue(name, irtype)
        self._symbol_map[symbol] = target

        # Initialize if a value is provided
        if node.value is not None:
            value = self.visit(node.value)
            self._emit_assign(value, target)

    def visit_AssignStatement(self, node: ast.AssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self._emit_assign(value, target)

    def _get_int_float_binop(self, op: ast.ArithmeticOp) -> tuple[
        type[ir.IntBinaryInstr], type[ir.FloatBinaryInstr]
    ]:
        return {
            ast.AddOp: (ir.IAdd, ir.FAdd),
            ast.SubOp: (ir.ISub, ir.FSub),
            ast.MulOp: (ir.IMul, ir.FMul),
            ast.DivOp: (ir.IDiv, ir.FDiv),
            ast.ModOp: (ir.IMod, ir.FMod),
        }[type(op)]
    
    def _get_cmp_op(self, op: ast.CompareOp) -> utils.CompareOp:
        return {
            ast.EqOp: utils.EqOp,
            ast.NeOp: utils.NeOp,
            ast.LtOp: utils.LtOp,
            ast.GtOp: utils.GtOp,
            ast.LeOp: utils.LeOp,
            ast.GeOp: utils.GeOp,
        }[type(op)]()

    def _get_float_from_int(self, value: ir.Value[ir.IntType]) -> ir.Value[ir.FloatType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.FloatType())
            self.emit(ir.IntToFloat(value, temp))
        elif isinstance(value, ir.Constant):
            temp = ir.Constant(float(value.value()), ir.FloatType())
        else:
            assert False
        return temp

    def visit_AugAssignStatement(self, node: ast.AugAssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)

        i_op, f_op = self._get_int_float_binop(node.op)

        if ir.int_typed(target) and ir.int_typed(value):
            # Both integers
            self.emit(i_op(target, value, target))
        elif ir.float_typed(target) and ir.float_typed(value):
            # Both floats
            self.emit(f_op(target, value, target))
        elif ir.float_typed(target) and ir.int_typed(value):
            # Float target, integer value - convert integer to float
            temp = self._get_float_from_int(value)
            self.emit(f_op(target, temp, target))
        else:
            assert False

    def visit_PassStatement(self, node: ast.PassStatement):
        pass

    def visit_ReturnStatement(self, node: ast.ReturnStatement):
        if node.value is None:
            self.emit(ir.Return())
        else:
            ret_val = self.visit(node.value)
            self.emit(ir.Return(ret_val))

    def visit_IfStatement(self, node: ast.IfStatement):
        # Evaluate the condition
        test = self.visit(node.test)
        # Ensure the condition is in a NamedValue (temporary or variable)
        if not isinstance(test, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self.emit(ir.IntAssign(test, temp))
            test = temp

        # Create basic blocks
        merge_block = self.new_block()  # Block where control flow merges
        if node.orelse:
            else_block = self.new_block()
        else:
            else_block = merge_block

        then_block = self.new_block()
        self.emit(ir.Branch(test, then_block, else_block))

        # Generate then branch
        self.builder.set_block(then_block)
        self.visit(node.body)
        if not self.is_terminated():
            self.emit(ir.Goto(merge_block))

        # Generate else branch if it exists
        if node.orelse:
            self.builder.set_block(else_block)
            self.visit(node.orelse)
            if not self.is_terminated():
                self.emit(ir.Goto(merge_block))

        # Continue with the merged block
        self.builder.set_block(merge_block)

    def visit_WhileStatement(self, node: ast.WhileStatement):
        # Basic blocks: condition check, loop body, and exit
        cond_block = self.new_block()
        body_block = self.new_block()
        end_block = self.new_block()

        # Push loop context for break/continue (break -> end_block, continue -> cond_block)
        self._loop_stack.append((end_block, cond_block))

        # Jump to condition check
        self.emit(ir.Goto(cond_block))
        self.builder.set_block(cond_block)

        if node.test is None:
            # Infinite loop
            self.builder.set_block(body_block)
            self.visit(node.body)
            self.emit(ir.Goto(body_block))
            self._loop_stack.pop()
            self.builder.set_block(end_block)
            return

        # Generate condition check
        test = self.visit(node.test)
        if not isinstance(test, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self._emit_assign(test, temp)
            test = temp
        self.emit(ir.Branch(test, body_block, end_block))

        # Generate loop body
        self.builder.set_block(body_block)
        self.visit(node.body)
        if not self.is_terminated():
            self.emit(ir.Goto(cond_block))  # Loop back

        # Pop loop context and set block to after the loop
        self._loop_stack.pop()
        self.builder.set_block(end_block)

    def visit_ForStatement(self, node: ast.ForStatement):
        with self.with_scope(node.scope):
            # Basic blocks: init, condition, body, increment, exit
            cond_block = self.new_block()
            body_block = self.new_block()
            update_block = self.new_block()
            end_block = self.new_block()

            # Push loop context (break -> end_block, continue -> update_block)
            self._loop_stack.append((end_block, update_block))

            # Execute initialization
            if node.init_stmt is not None:
                self.visit(node.init_stmt)
            self.emit(ir.Goto(cond_block))
            self.builder.set_block(cond_block)

            # Check condition
            if node.test is None:
                # No condition means infinite loop
                self.builder.set_block(body_block)
                self.visit(node.body)
                if node.end_stmt is not None:
                    self.visit(node.end_stmt)
                self.emit(ir.Goto(body_block))
                self._loop_stack.pop()
                self.builder.set_block(end_block)
                return
            
            test = self.visit(node.test)
            if not isinstance(test, ir.NamedValue):
                temp = self.builder.new_temp(ir.IntType())
                self._emit_assign(test, temp)
                test = temp
            self.emit(ir.Branch(test, body_block, end_block))

            # Execute loop body
            self.builder.set_block(body_block)
            self.visit(node.body)
            if not self.is_terminated():
                self.emit(ir.Goto(update_block))

            # Execute increment
            self.builder.set_block(update_block)
            if node.end_stmt is not None:
                self.visit(node.end_stmt)
            self.emit(ir.Goto(cond_block))

            # Pop loop context
            self._loop_stack.pop()
            self.builder.set_block(end_block)

    def visit_BreakStatement(self, node: ast.BreakStatement):
        if not self._loop_stack:
            return
        self.emit(ir.Goto(self._loop_stack[-1][0]))  # Jump to break block

    def visit_ContinueStatement(self, node: ast.ContinueStatement):
        if not self._loop_stack:
            return
        self.emit(ir.Goto(self._loop_stack[-1][1]))  # Jump to continue block

    def _with_expression(self, node: ast.Expression):
        if node.evaluation_scope is not None:
            return self.with_scope(node.evaluation_scope)
        return contextmanager(lambda: (yield))()  # No-op context manager

    def visit_BinaryExpression(self, node: ast.BinaryExpression) -> ir.Value:
        with self._with_expression(node):
            lhs = self.visit(node.lhs)
            rhs = self.visit(node.rhs)
            target_type = self._get_irtype(node.type_info)
            target = self.builder.new_temp(target_type)

            if isinstance(node.op, ast.ArithmeticOp):
                # Arithmetic operations (+, -, *, /, %)
                i_op, f_op = self._get_int_float_binop(node.op)

                if ir.int_typed(lhs) and ir.int_typed(rhs):
                    # Integer arithmetic
                    assert ir.int_typed(target)
                    self.emit(i_op(lhs, rhs, target))

                elif ir.float_typed(lhs) and ir.float_typed(rhs):
                    # Float arithmetic
                    target = cast(ir.NamedValue[ir.FloatType], target)
                    self.emit(f_op(lhs, rhs, target))

                # Mixed types - convert integer(s) to float
                elif ir.int_typed(lhs) and ir.float_typed(rhs):
                    lhs = self._get_float_from_int(lhs)
                    assert ir.float_typed(target)
                    self.emit(f_op(lhs, rhs, target))

                elif ir.float_typed(lhs) and ir.int_typed(rhs):
                    rhs = self._get_float_from_int(rhs)
                    assert ir.float_typed(target)
                    self.emit(f_op(lhs, rhs, target))
                
                else:
                    assert False

            elif isinstance(node.op, ast.CompareOp):
                # Comparison operations (==, !=, <, >, <=, >=)
                op = self._get_cmp_op(node.op)

                if ir.int_typed(lhs) and ir.int_typed(rhs):
                    # Integer comparison
                    self.emit(ir.ICmp(lhs, rhs, target, op))
                
                elif ir.float_typed(lhs) and ir.float_typed(rhs):
                    self.emit(ir.FCmp(lhs, rhs, target, op))

                elif ir.int_typed(lhs) and ir.float_typed(rhs):
                    # Float comparison with possible integer promotion
                    lhs = self._get_float_from_int(lhs)
                    self.emit(ir.FCmp(lhs, rhs, target, op))
                
                elif ir.float_typed(lhs) and ir.int_typed(rhs):
                    rhs = self._get_float_from_int(rhs)
                    self.emit(ir.FCmp(lhs, rhs, target, op))
                
                else:
                    assert False

            elif isinstance(node.op, ast.BinaryBoolOp):
                assert ir.int_typed(target)
                # Logical operations (and, or)
                if isinstance(node.op, ast.AndOp):
                    self.emit(ir.And(lhs, rhs, target))
                else:  # OrOp
                    self.emit(ir.Or(lhs, rhs, target))
            else:
                raise NotImplementedError(f"Unsupported binary operator: {node.op}")

            return target

    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> ir.Value:
        with self._with_expression(node):
            operand = self.visit(node.operand)
            target_type = self._get_irtype(node.type_info)
            target = self.builder.new_temp(target_type)

            if isinstance(node.op, ast.NotOp):
                # Logical NOT
                assert ir.int_typed(target)
                assert isinstance(operand, ir.NamedValue)  # Constants are already folded
                self.emit(ir.Not(operand, target))
            
            elif isinstance(node.op, ast.NegOp):
                # Arithmetic negation (unary minus)
                if ir.int_typed(operand) and ir.int_typed(target):
                    # Negate integer: 0 - operand
                    zero = ir.Constant(0, ir.IntType())
                    self.emit(ir.ISub(zero, operand, target))
                elif ir.float_typed(operand) and ir.float_typed(target):
                    # Negate float: 0.0 - operand
                    zero = ir.Constant(0.0, ir.FloatType())
                    self.emit(ir.FSub(zero, operand, target))
                else:
                    assert False

            elif isinstance(node.op, ast.PosOp):
                # Unary plus does nothing, just copy the value
                self._emit_assign(operand, target)
            else:
                raise NotImplementedError(f"Unary operator {node.op}")

            return target

    def visit_CallExpression(self, node: ast.CallExpression) -> ir.Value | None:
        with self._with_expression(node):
            callee = node.callee
            assert isinstance(callee, ast.Identifier)
            func_name = callee.name

            # Look up the IR function
            ir_func = self._func_map.get(func_name)
            assert ir_func is not None

            # Get argument bindings from the semantic analysis
            bound = self.call_param_bindings.get(node)
            assert bound is not None
            signature = self.symtab.lookup_func_signature(func_name)
            assert signature is not None
            
            # Evaluate arguments in the correct order (already bound by semantic analyzer)
            args = [self.visit(bound[p.name]) for p in signature.param_infos]

            # Generate the call instruction
            return_type = ir_func.type.return_type
            if isinstance(return_type, ir.VoidType):
                # Void function - no return value
                self.emit(ir.Call(ir_func, args))
                return None
            else:
                # Non-void function - store result in a temporary
                target = self.builder.new_temp(return_type)
                self.emit(ir.Call(ir_func, args, target))
                return target

    def visit_Constant(self, node: ast.Constant) -> ir.Constant:
        with self._with_expression(node):
            value = node.value
            if isinstance(node.type, ast.IntType):
                return ir.Constant(value, ir.IntType())
            elif isinstance(node.type, ast.FloatType):
                return ir.Constant(value, ir.FloatType())
            elif isinstance(node.type, ast.BoolType):
                # Booleans as integers (1 for true, 0 for false)
                return ir.Constant(int(value), ir.IntType())
            else:
                assert False

    def visit_Identifier(self, node: ast.Identifier) -> ir.NamedValue:
        with self._with_expression(node):
            name = node.name
            symbol = self.symtab.lookup(name)
            assert symbol is not None, f"Identifier '{name}' not in symbol table"
            ir_val = self._symbol_map.get(symbol)
            assert ir_val is not None, f"Identifier '{name}' has no IR mapping"
            return ir_val

    def generic_visit(self, node: ast.ASTNode, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.{type(node).__name__}() not implemented")


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
