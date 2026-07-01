from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import overload

from vcs import ast
from vcs import ir
from vcs import semantic as sem
from vcs import utils


@dataclass
class LoopContext:
    """Context information for a loop (break and continue targets)."""
    break_block: ir.BasicBlock
    continue_block: ir.BasicBlock


class TypeCoercer:
    """
    Handles type promotion and conversion between different numeric types.
    Centralizes all type conversion logic to avoid duplication.
    """
    
    def __init__(self, builder: ir.IRBuilder, emitter):
        self.builder = builder
        self._emit = emitter
    
    def promote_to_float(self, value: ir.Value) -> ir.Value[ir.FloatType]:
        """Promote a value to float type, inserting conversion instructions if needed."""
        if ir.float_typed(value):
            return value
        elif ir.int_typed(value):
            return self._int_to_float(value)
        elif ir.fixed_typed(value):
            return self._fixed_to_float(value)
        else:
            raise TypeError(f"Cannot promote {type(value)} to float")
    
    def promote_to_fixed(self, value: ir.Value) -> ir.Value[ir.FixedType]:
        """Promote a value to fixed-point type."""
        if ir.fixed_typed(value):
            return value
        elif ir.int_typed(value):
            return self._int_to_fixed(value)
        elif ir.float_typed(value):
            return self._float_to_fixed(value)
        else:
            raise TypeError(f"Cannot promote {type(value)} to fixed")
    
    def promote_to_int(self, value: ir.Value) -> ir.Value[ir.IntType]:
        """Promote a value to integer type."""
        if ir.int_typed(value):
            return value
        elif ir.float_typed(value):
            return self._float_to_int(value)
        elif ir.fixed_typed(value):
            return self._fixed_to_int(value)
        else:
            raise TypeError(f"Cannot promote {type(value)} to int")
    
    def unify_pair(self, lhs: ir.Value, rhs: ir.Value) -> tuple[ir.Value, ir.Value, ir.Type]:
        """
        Unify two values to a common type, inserting conversions as needed.
        Returns (promoted_lhs, promoted_rhs, common_type).
        """
        if ir.float_typed(lhs) or ir.float_typed(rhs):
            # Float has highest precedence
            target_type = ir.FloatType()
            return (
                self.promote_to_float(lhs),
                self.promote_to_float(rhs),
                target_type
            )
        elif ir.fixed_typed(lhs) or ir.fixed_typed(rhs):
            # Fixed-point is second precedence
            target_type = ir.FixedType()
            return (
                self.promote_to_fixed(lhs),
                self.promote_to_fixed(rhs),
                target_type
            )
        else:
            # Both are integers
            target_type = ir.IntType()
            return (lhs, rhs, target_type)
    
    def _int_to_float(self, value: ir.Value[ir.IntType]) -> ir.Value[ir.FloatType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.FloatType())
            self._emit(ir.IntToFloat(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(float(value.int_value()), ir.FloatType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from int to float")
    
    def _fixed_to_float(self, value: ir.Value[ir.FixedType]) -> ir.Value[ir.FloatType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.FloatType())
            self._emit(ir.FixedToFloat(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(float(value.value()), ir.FloatType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from fixed to float")
    
    def _int_to_fixed(self, value: ir.Value[ir.IntType]) -> ir.Value[ir.FixedType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.FixedType())
            self._emit(ir.IntToFixed(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(float(value.int_value()), ir.FixedType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from int to fixed")
    
    def _float_to_fixed(self, value: ir.Value[ir.FloatType]) -> ir.Value[ir.FixedType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.FixedType())
            self._emit(ir.FloatToFixed(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(float(value.float_value()), ir.FixedType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from float to fixed")
    
    def _float_to_int(self, value: ir.Value[ir.FloatType]) -> ir.Value[ir.IntType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self._emit(ir.FloatToInt(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(int(value.float_value()), ir.IntType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from float to int")
    
    def _fixed_to_int(self, value: ir.Value[ir.FixedType]) -> ir.Value[ir.IntType]:
        if isinstance(value, ir.NamedValue):
            temp = self.builder.new_temp(ir.IntType())
            self._emit(ir.FixedToInt(value, temp))
            return temp
        elif isinstance(value, ir.Constant):
            return ir.Constant(int(value.value()), ir.IntType())
        else:
            raise TypeError(f"Cannot convert {type(value)} from fixed to int")


class BinaryOpMapper:
    """
    Maps AST binary operations to IR instruction classes based on operand types.
    Uses a lookup table to eliminate repetitive type-checking code.
    """
    
    def __init__(self, coercer: TypeCoercer):
        self.coercer = coercer
        
        # Maps (operation_type, operand_type_tuple) -> instruction_class
        self._arith_map: dict[
            tuple[type[ast.ArithmeticOp], tuple[type[ir.Type], type[ir.Type]]], type[ir.BinaryInstr]
        ] = {
            (ast.AddOp, (ir.IntType, ir.IntType)): ir.IAdd,
            (ast.AddOp, (ir.FloatType, ir.FloatType)): ir.FAdd,
            (ast.AddOp, (ir.FixedType, ir.FixedType)): ir.XAdd,
            
            (ast.SubOp, (ir.IntType, ir.IntType)): ir.ISub,
            (ast.SubOp, (ir.FloatType, ir.FloatType)): ir.FSub,
            (ast.SubOp, (ir.FixedType, ir.FixedType)): ir.XSub,
            
            (ast.MulOp, (ir.IntType, ir.IntType)): ir.IMul,
            (ast.MulOp, (ir.FloatType, ir.FloatType)): ir.FMul,
            (ast.MulOp, (ir.FixedType, ir.FixedType)): ir.XMul,
            
            (ast.DivOp, (ir.IntType, ir.IntType)): ir.IDiv,
            (ast.DivOp, (ir.FloatType, ir.FloatType)): ir.FDiv,
            (ast.DivOp, (ir.FixedType, ir.FixedType)): ir.XDiv,
            
            (ast.ModOp, (ir.IntType, ir.IntType)): ir.IMod,
            (ast.ModOp, (ir.FloatType, ir.FloatType)): ir.FMod,
            (ast.ModOp, (ir.FixedType, ir.FixedType)): ir.XMod,
        }
        
        self._cmp_map: dict[
            tuple[type[ast.CompareOp], tuple[type[ir.Type], type[ir.Type]]], type[ir.CmpInstr]
        ] = {
            (ast.EqOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.EqOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.EqOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
            
            (ast.NeOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.NeOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.NeOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
            
            (ast.LtOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.LtOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.LtOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
            
            (ast.GtOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.GtOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.GtOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
            
            (ast.LeOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.LeOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.LeOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
            
            (ast.GeOp, (ir.IntType, ir.IntType)): ir.ICmp,
            (ast.GeOp, (ir.FloatType, ir.FloatType)): ir.FCmp,
            (ast.GeOp, (ir.FixedType, ir.FixedType)): ir.XCmp,
        }
    
    def get_arith_op(
        self, op: ast.ArithmeticOp, lhs: ir.Value, rhs: ir.Value
    ) -> tuple[type[ir.BinaryInstr], ir.Value, ir.Value, ir.Type]:
        """
        Get the appropriate arithmetic instruction class and promoted operands.
        Returns (instruction_class, promoted_lhs, promoted_rhs).
        """
        lhs_p, rhs_p, target_type = self.coercer.unify_pair(lhs, rhs)
        key = (type(op), (type(lhs_p.type), type(rhs_p.type)))
        instr_cls = self._arith_map.get(key)
        if instr_cls is None:
            raise NotImplementedError(f"No arithmetic op mapping for {op} on {type(lhs_p.type)} and {type(rhs_p.type)}")
        return instr_cls, lhs_p, rhs_p, target_type
    
    def get_cmp_op(
        self, op: ast.CompareOp, lhs: ir.Value, rhs: ir.Value
    ) -> tuple[type[ir.CmpInstr], ir.Value, ir.Value]:
        """
        Get the appropriate comparison instruction class and promoted operands.
        Returns (instruction_class, promoted_lhs, promoted_rhs).
        """
        lhs_p, rhs_p, target_type = self.coercer.unify_pair(lhs, rhs)
        key = (type(op), (type(lhs_p.type), type(rhs_p.type)))
        instr_cls = self._cmp_map.get(key)
        if instr_cls is None:
            raise NotImplementedError(f"No comparison op mapping for {op} on {type(lhs_p.type)} and {type(rhs_p.type)}")
        return instr_cls, lhs_p, rhs_p
    
    @staticmethod
    def get_cmp_operator(op: ast.CompareOp) -> utils.CompareOp:
        """Convert AST comparison operator to IR comparison operator."""
        mapping = {
            ast.EqOp: utils.EqOp,
            ast.NeOp: utils.NeOp,
            ast.LtOp: utils.LtOp,
            ast.GtOp: utils.GtOp,
            ast.LeOp: utils.LeOp,
            ast.GeOp: utils.GeOp,
        }
        return mapping[type(op)]()
    
    @staticmethod
    def get_aug_arith_op(op: ast.ArithmeticOp) -> tuple[
        type[ir.IntBinaryInstr], type[ir.FloatBinaryInstr], type[ir.FixedBinaryInstr]
    ]:
        """Get the three arithmetic instruction types for augmented assignment."""
        mapping = {
            ast.AddOp: (ir.IAdd, ir.FAdd, ir.XAdd),
            ast.SubOp: (ir.ISub, ir.FSub, ir.XSub),
            ast.MulOp: (ir.IMul, ir.FMul, ir.XMul),
            ast.DivOp: (ir.IDiv, ir.FDiv, ir.XDiv),
            ast.ModOp: (ir.IMod, ir.FMod, ir.XMod),
        }
        return mapping[type(op)]


class ScopeManager:
    """
    Manages symbol scopes and temporary variables during IR generation.
    Encapsulates scope-related logic to keep IRGenerator focused on IR generation.
    """
    
    def __init__(self, symtab: sem.SymbolTable):
        self.symtab = symtab
        self._extended_locals: dict[sem.Scope, list[ir.NamedValue]] = {}
        self._symbol_map: dict[sem.Symbol, ir.NamedValue] = {}
    
    def enter_scope(self, scope: sem.Scope):
        """Enter a new scope."""
        return self.symtab.with_scope(scope)
    
    def get_current_scope(self) -> sem.Scope | None:
        """Get the current scope."""
        return self.symtab.cur_scope
    
    def lookup_symbol(self, name: str) -> sem.Symbol | None:
        """Look up a symbol in the symbol table."""
        return self.symtab.lookup(name)
    
    def get_symbol_ir(self, symbol: sem.Symbol) -> ir.NamedValue | None:
        """Get the IR value associated with a symbol."""
        return self._symbol_map.get(symbol)
    
    def set_symbol_ir(self, symbol: sem.Symbol, value: ir.NamedValue):
        """Associate a symbol with its IR value."""
        self._symbol_map[symbol] = value
    
    def add_temp_var(self, var: ir.NamedValue):
        """Add a temporary variable to the current scope."""
        cur_scope = self.symtab.cur_scope
        if cur_scope is None or cur_scope is self.symtab.global_scope:
            return  # Don't track temporaries in global scope
        
        if cur_scope not in self._extended_locals:
            self._extended_locals[cur_scope] = []
        self._extended_locals[cur_scope].append(var)
    
    def get_all_locals(self) -> list[ir.NamedValue]:
        """Get all local variables in the current scope chain."""
        local_symbols = self.symtab.get_locals()
        result = [self._symbol_map[sym] for sym in local_symbols if sym in self._symbol_map]
        
        scope = self.symtab.cur_scope
        while scope and scope is not self.symtab.global_scope:
            if scope in self._extended_locals:
                result.extend(self._extended_locals[scope])
            scope = scope.enclosing
        return result


class RecursiveCallTransformer:
    """
    Transforms function calls in recursive functions by adding Push/Pop
    instructions around each call to save/restore local variables.
    This is a separate pass that runs after IR generation.
    """
    
    def __init__(self, module: ir.Module, call_locals_map: dict[ir.Call, list[ir.NamedValue]]):
        self.module = module
        self.call_locals_map = call_locals_map
        self.builder = ir.IRBuilder()
    
    def transform(self):
        """Apply the transformation to all functions in the module."""
        if not self.call_locals_map:
            return
        
        call_graph = self.module.build_call_graph()
        
        for func in self.module.functions:
            if call_graph.is_recursive(func):
                self._transform_function(func)
    
    def _transform_function(self, func: ir.Function):
        """Transform a recursive function by wrapping calls with Push/Pop."""
        for block in func.blocks:
            self.builder.set_block(block)
            i = 0
            while i < len(block.instructions):
                inst = block.instructions[i]
                if isinstance(inst, ir.Call):
                    local_vars = self.call_locals_map.get(inst, [])
                    if local_vars:
                        # Insert Push before the call
                        self.builder.insert(i, ir.Push(local_vars))
                        i += 1
                        # Insert Pop after the call
                        self.builder.insert(i + 1, ir.Pop(local_vars))
                i += 1


class IRGenerator(ast.ASTNodeVisitor):
    """
    Generates IR from AST.
    Refactored version with separated concerns and reduced duplication.
    """
    
    def __init__(
        self,
        namespace: str,
        typechecker: sem.SemanticAnalyzer,
        lineno_digits: int = 5,  # Default max digits for line numbers in block names
        dump_ir=False,
        dump_cg=False,
    ):
        self.namespace = namespace
        self.typechecker = typechecker
        self.lineno_digits = lineno_digits

        # Debug flags
        self.dump_ir = dump_ir
        self.dump_cg = dump_cg

        # IR module and builder
        self.module = ir.Module(namespace)
        self.builder = ir.IRBuilder()
        self.cur_func: ir.Function | None = None
        
        # Function name -> IR function mapping
        self._func_map: dict[str, ir.Function] = {}
        
        # Loop stack for break/continue
        self._loop_stack: list[LoopContext] = []
        
        # Call instruction -> local variables at call site
        self._call_locals_map: dict[ir.Call, list[ir.NamedValue]] = {}
        
        # Managed components
        self.scope_manager = ScopeManager(typechecker.symtab)
        self.coercer = TypeCoercer(self.builder, self._emit)
        self.op_mapper = BinaryOpMapper(self.coercer)
        
        # Bind call_param_bindings from typechecker
        self.call_param_bindings = typechecker.call_param_bindings
    
    def generate(self) -> ir.Module | None:
        """
        Run the semantic analysis and generate IR.
        """
        tree = self.typechecker.analyze()
        if tree is None:
            return None
        
        self._declare_functions(tree)
        self.visit(tree)
        self._post_process_calls()

        if self.dump_ir:
            utils.print_info(str(self.module) + "\n")

        if self.dump_cg:
            call_graph = self.module.build_call_graph()
            for src, dsts in call_graph.func_calls.items():
                utils.print_info(f"{src.name} -> {', '.join(dst.name for dst in dsts)}")
        
        return self.module
    
    def _emit(self, inst: ir.Instruction):
        """Emit an instruction to the current basic block."""
        self.builder.emit(inst)
    
    def is_terminated(self) -> bool:
        """Check if the current basic block is terminated."""
        return self.builder.is_terminated()
    
    def get_irtype(self, semtype: sem.TypeInfo) -> ir.Type:
        """Convert a semantic type to an IR type."""
        return self.builder.get_irtype(semtype)
    
    def _get_irfunc(self, name: str, signature: sem.FunctionTypeInfo) -> ir.Function:
        """Create an IR function from a semantic function signature."""
        return_type = self.get_irtype(signature.return_type)
        param_types = [self.get_irtype(p.type_info) for p in signature.param_infos]
        param_names = [p.name for p in signature.param_infos]
        func_type = ir.FunctionType(return_type, param_types)
        return ir.Function(name, func_type, param_names)
    
    def new_block(self, label: str, lineno: int) -> ir.BasicBlock:
        """Create a new basic block in the current function."""
        assert self.cur_func is not None
        block_name = f"{label}_{str(lineno).zfill(self.lineno_digits)}"

        i = 1
        new_block_name = block_name
        while self.cur_func.has_block(new_block_name):
            new_block_name = f"{block_name}_{i}"
            i += 1
        block_name = new_block_name

        return self.cur_func.create_block(block_name)
    
    def _ensure_named_value(self, value: ir.Value) -> ir.NamedValue:
        """Ensure a value is in a NamedValue (temporary or variable)."""
        if isinstance(value, ir.NamedValue):
            return value
        temp = self.builder.new_temp(value.type)
        self.scope_manager.add_temp_var(temp)
        self._emit_assign(value, temp)
        return temp
    
    def _emit_assign[T: ir.Type](self, value: ir.Value[T], target: ir.NamedValue[T]):
        """Emit an assignment instruction based on the types involved."""
        if ir.int_typed(value) and ir.int_typed(target):
            self._emit(ir.IntAssign(value, target))
        elif ir.fixed_typed(value) and ir.fixed_typed(target):
            self._emit(ir.FixedAssign(value, target))
        elif ir.float_typed(value) and ir.float_typed(target):
            self._emit(ir.FloatAssign(value, target))
        else:
            # Try to coerce
            if ir.int_typed(value) and ir.float_typed(target):
                converted = self.coercer.promote_to_float(value)
                self._emit(ir.FloatAssign(converted, target))
            elif ir.fixed_typed(value) and ir.float_typed(target):
                converted = self.coercer.promote_to_float(value)
                self._emit(ir.FloatAssign(converted, target))
            elif ir.float_typed(value) and ir.int_typed(target):
                converted = self.coercer.promote_to_int(value)
                self._emit(ir.IntAssign(converted, target))
            elif ir.fixed_typed(value) and ir.int_typed(target):
                converted = self.coercer.promote_to_int(value)
                self._emit(ir.IntAssign(converted, target))
            else:
                raise TypeError(f"Cannot assign {type(value.type)} to {type(target.type)}")
    
    def _declare_functions(self, node: ast.Module):
        """First pass: create function declarations and map parameters."""
        for sub in node.body:
            if isinstance(sub, ast.FunctionDeclaration):
                func = self._get_irfunc(sub.name, sub.signature)
                self.module.add_func(func)
                self._func_map[sub.name] = func
                
                # Map function parameters to IR values
                with self.scope_manager.enter_scope(sub.scope):
                    for param_info in sub.signature.param_infos:
                        symbol = self.scope_manager.lookup_symbol(param_info.name)
                        assert symbol is not None, f"Parameter {param_info.name} not found"
                        irtype = self.get_irtype(param_info.type_info)
                        param_val = ir.NamedValue(param_info.name, irtype)
                        self.scope_manager.set_symbol_ir(symbol, param_val)
    
    def _post_process_calls(self):
        """Third pass: transform call instructions for recursive functions."""
        transformer = RecursiveCallTransformer(self.module, self._call_locals_map)
        transformer.transform()
    
    def visit_Module(self, node: ast.Module):
        """Second pass: generate function bodies."""
        for sub in node.body:
            self.visit(sub)
    
    def visit_FunctionDeclaration(self, node: ast.FunctionDeclaration):
        """Generate IR for a function body."""
        func = self._func_map[node.name]
        self.cur_func = func
        self._block_id = 0
        self.builder.reset_value_counter()
        self.builder.set_block(func.entry_block)
        
        with self.scope_manager.enter_scope(node.scope):
            if node.body is not None:
                self.visit(node.body)
        
        # Add implicit return if needed
        if not self.is_terminated():
            if isinstance(node.signature.return_type, sem.VoidTypeInfo):
                self._emit(ir.Return())
            # Non-void functions without return will have errors elsewhere
    
    def visit_Comment(self, node: ast.Comment):
        self._emit(ir.Comment(node.value))
    
    def visit_Block(self, node: ast.Block):
        with self.scope_manager.enter_scope(node.scope):
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_ExpressionStatement(self, node: ast.ExpressionStatement):
        self.visit(node.expression)
    
    def visit_VariableDeclarationStatement(self, node: ast.VariableDeclarationStatement):
        name = node.name
        symbol = self.scope_manager.lookup_symbol(name)
        assert symbol is not None
        irtype = self.get_irtype(symbol.type_info)
        
        target = ir.NamedValue(name, irtype)
        self.scope_manager.set_symbol_ir(symbol, target)
        
        if node.value is not None:
            value = self.visit(node.value)
            self._emit_assign(value, target)
    
    def visit_AssignStatement(self, node: ast.AssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self._emit_assign(value, target)
    
    def visit_AugAssignStatement(self, node: ast.AugAssignStatement):
        target = self.visit(node.target)
        value = self.visit(node.value)
        
        # Get instruction types for this operation
        i_op, f_op, x_op = BinaryOpMapper.get_aug_arith_op(node.op)
        
        # Unify types and determine operation
        if ir.int_typed(target) and ir.int_typed(value):
            self._emit(i_op(target, value, target))
        elif ir.float_typed(target) and ir.float_typed(value):
            self._emit(f_op(target, value, target))
        elif ir.fixed_typed(target) and ir.fixed_typed(value):
            self._emit(x_op(target, value, target))
        elif ir.float_typed(target) and ir.int_typed(value):
            converted = self.coercer.promote_to_float(value)
            self._emit(f_op(target, converted, target))
        elif ir.float_typed(target) and ir.fixed_typed(value):
            converted = self.coercer.promote_to_float(value)
            self._emit(f_op(target, converted, target))
        elif ir.int_typed(target) and ir.float_typed(value):
            converted = self.coercer.promote_to_int(value)
            self._emit(i_op(target, converted, target))
        elif ir.fixed_typed(target) and ir.float_typed(value):
            converted = self.coercer.promote_to_fixed(value)
            self._emit(x_op(target, converted, target))
        else:
            raise TypeError(f"Incompatible types for augmented assignment: {type(target.type)} and {type(value.type)}")
    
    def visit_PassStatement(self, node: ast.PassStatement):
        pass
    
    def visit_TellStatement(self, node: ast.TellStatement):
        self._emit(ir.Tell(self.visit(node.value)))
    
    def visit_ReturnStatement(self, node: ast.ReturnStatement):
        if node.value is None:
            self._emit(ir.Return())
        else:
            ret_val = self.visit(node.value)
            self._emit(ir.Return(ret_val))
    
    def visit_IfStatement(self, node: ast.IfStatement):
        test = self._ensure_named_value(self.visit(node.test))
        
        body_block = self.new_block("if", node.body.lineno)
        end_block = self.new_block("endif", node.end_lineno + 1)
        else_block = self.new_block("else", node.orelse.lineno) if node.orelse else end_block
        
        self._emit(ir.Branch(test, body_block, else_block))
        
        # Generate then branch
        self.builder.set_block(body_block)
        self.visit(node.body)
        if not self.is_terminated():
            self._emit(ir.Goto(end_block))
        
        # Generate else branch if it exists
        if node.orelse:
            self.builder.set_block(else_block)
            self.visit(node.orelse)
            if not self.is_terminated():
                self._emit(ir.Goto(end_block))
        
        self.builder.set_block(end_block)
    
    def visit_WhileStatement(self, node: ast.WhileStatement):
        cond_block = self.new_block("continue", node.lineno)
        body_block = self.new_block("while", node.body.lineno)
        end_block = self.new_block("break", node.end_lineno + 1)
        
        self._loop_stack.append(LoopContext(end_block, cond_block))
        self._emit(ir.Goto(cond_block))
        
        if node.test is None:
            # Infinite loop
            self.builder.set_block(body_block)
            self.visit(node.body)
            if not self.is_terminated():
                self._emit(ir.Goto(body_block))
        else:
            self.builder.set_block(cond_block)
            test = self._ensure_named_value(self.visit(node.test))
            self._emit(ir.Branch(test, body_block, end_block))
            
            self.builder.set_block(body_block)
            self.visit(node.body)
            if not self.is_terminated():
                self._emit(ir.Goto(cond_block))
        
        self._loop_stack.pop()
        self.builder.set_block(end_block)
    
    def visit_ForStatement(self, node: ast.ForStatement):
        with self.scope_manager.enter_scope(node.scope):
            cond_block = self.new_block("continue", node.lineno)
            body_block = self.new_block("for", node.body.lineno)
            end_block = self.new_block("break", node.end_lineno + 1)
            
            self._loop_stack.append(LoopContext(end_block, cond_block))
            
            if node.init_stmt is not None:
                self.visit(node.init_stmt)
            
            if node.test is None:
                # Infinite loop
                self.builder.set_block(body_block)
                self.visit(node.body)
                if node.end_stmt is not None:
                    self.visit(node.end_stmt)
                if not self.is_terminated():
                    self._emit(ir.Goto(body_block))
            else:
                self._emit(ir.Goto(cond_block))
                self.builder.set_block(cond_block)

                test = self._ensure_named_value(self.visit(node.test))
                self._emit(ir.Branch(test, body_block, end_block))
                
                self.builder.set_block(body_block)
                self.visit(node.body)

                if node.end_stmt is not None:
                    self.visit(node.end_stmt)
                self._emit(ir.Goto(cond_block))
            
            self._loop_stack.pop()
            self.builder.set_block(end_block)
    
    def visit_BreakStatement(self, node: ast.BreakStatement):
        if self._loop_stack:
            self._emit(ir.Goto(self._loop_stack[-1].break_block))
    
    def visit_ContinueStatement(self, node: ast.ContinueStatement):
        if self._loop_stack:
            self._emit(ir.Goto(self._loop_stack[-1].continue_block))
    
    def _with_expression(self, node: ast.Expression):
        """Context manager for expression evaluation scope."""
        if node.evaluation_scope is not None:
            return self.scope_manager.enter_scope(node.evaluation_scope)
        return contextmanager(lambda: (yield))()
    
    def visit_BinaryExpression(self, node: ast.BinaryExpression) -> ir.Value:
        with self._with_expression(node):
            lhs = self.visit(node.lhs)
            rhs = self.visit(node.rhs)
            
            if isinstance(node.op, ast.ArithmeticOp):
                instr_cls, lhs_p, rhs_p, target_type = self.op_mapper.get_arith_op(node.op, lhs, rhs)
                target = self.builder.new_temp(target_type)
                self.scope_manager.add_temp_var(target)
                self._emit(instr_cls(lhs_p, rhs_p, target))  # type: ignore
                return target
            
            elif isinstance(node.op, ast.CompareOp):
                instr_cls, lhs_p, rhs_p = self.op_mapper.get_cmp_op(node.op, lhs, rhs)
                target = self.builder.new_temp(ir.IntType())
                self.scope_manager.add_temp_var(target)
                cmp_op = BinaryOpMapper.get_cmp_operator(node.op)
                self._emit(instr_cls(lhs_p, rhs_p, target, cmp_op))
                return target
            
            elif isinstance(node.op, ast.BinaryBoolOp):
                lhs = self._ensure_named_value(lhs)
                rhs = self._ensure_named_value(rhs)
                target = self.builder.new_temp(ir.IntType())
                self.scope_manager.add_temp_var(target)
                if isinstance(node.op, ast.AndOp):
                    self._emit(ir.And(lhs, rhs, target))
                elif isinstance(node.op, ast.OrOp):
                    self._emit(ir.Or(lhs, rhs, target))
                else:
                    raise NotImplementedError(f"Unsupported binary boolean operator: {node.op}")
                return target
            
            else:
                raise NotImplementedError(f"Unsupported binary operator: {node.op}")
    
    def visit_UnaryExpression(self, node: ast.UnaryExpression) -> ir.Value:
        with self._with_expression(node):
            operand = self.visit(node.operand)
            
            if isinstance(node.op, ast.NotOp):
                operand = self._ensure_named_value(operand)
                target = self.builder.new_temp(ir.IntType())
                self.scope_manager.add_temp_var(target)
                self._emit(ir.Not(operand, target))
                return target
            
            elif isinstance(node.op, ast.NegOp):
                # Unify to appropriate type and negate with zero
                if ir.int_typed(operand):
                    target_type = ir.IntType()
                    zero = ir.Constant(0, ir.IntType())
                    target = self.builder.new_temp(ir.IntType())
                    self.scope_manager.add_temp_var(target)
                    self._emit(ir.ISub(zero, operand, target))
                elif ir.fixed_typed(operand):
                    target_type = ir.FixedType()
                    zero = ir.Constant(0.0, ir.FixedType())
                    target = self.builder.new_temp(ir.FixedType())
                    self.scope_manager.add_temp_var(target)
                    self._emit(ir.XSub(zero, operand, target))
                elif ir.float_typed(operand):
                    target_type = ir.FloatType()
                    zero = ir.Constant(0.0, ir.FloatType())
                    target = self.builder.new_temp(ir.FloatType())
                    self.scope_manager.add_temp_var(target)
                    self._emit(ir.FSub(zero, operand, target))
                else:
                    raise TypeError(f"Cannot negate type {type(operand.type)}")
                return target
            
            elif isinstance(node.op, ast.PosOp):
                # Unary plus is a no-op; just copy the value
                target_type = operand.type
                target = self.builder.new_temp(target_type)
                self.scope_manager.add_temp_var(target)
                self._emit_assign(operand, target)
                return target
            
            else:
                raise NotImplementedError(f"Unary operator {node.op}")
    
    def visit_CallExpression(self, node: ast.CallExpression) -> ir.Value | None:
        with self._with_expression(node):
            callee = node.callee
            assert isinstance(callee, ast.Identifier)
            func_name = callee.name
            
            ir_func = self._func_map.get(func_name)
            assert ir_func is not None
            
            bound = self.call_param_bindings.get(node)
            assert bound is not None
            signature = self.scope_manager.lookup_symbol(func_name)
            assert signature is not None
            # signature is a FunctionSymbol, get its type info
            func_sig = signature.type_info
            assert isinstance(func_sig, sem.FunctionTypeInfo)
            
            args = {p.name: self.visit(bound[p.name]) for p in func_sig.param_infos}
            
            return_type = ir_func.type.return_type
            if isinstance(return_type, ir.VoidType):
                inst = ir.Call(ir_func, args)
                self._emit(inst)
                self._call_locals_map[inst] = self.scope_manager.get_all_locals()
                return None
            else:
                target = self.builder.new_temp(return_type)
                inst = ir.Call(ir_func, args, target)
                self._call_locals_map[inst] = self.scope_manager.get_all_locals()
                self.scope_manager.add_temp_var(target)
                self._emit(inst)
                return target
    
    def visit_Constant(self, node: ast.Constant) -> ir.Constant:
        with self._with_expression(node):
            value = node.value
            if isinstance(node.type, ast.IntType):
                return ir.Constant(value, ir.IntType())
            elif isinstance(node.type, ast.FixedType):
                return ir.Constant(value, ir.FixedType())
            elif isinstance(node.type, ast.FloatType):
                return ir.Constant(value, ir.FloatType())
            elif isinstance(node.type, ast.BoolType):
                return ir.Constant(int(value), ir.IntType())
            else:
                raise TypeError(f"Unsupported constant type: {type(node.type)}")
    
    def visit_Identifier(self, node: ast.Identifier) -> ir.NamedValue:
        with self._with_expression(node):
            name = node.name
            symbol = self.scope_manager.lookup_symbol(name)
            assert symbol is not None, f"Identifier '{name}' not in symbol table"
            ir_val = self.scope_manager.get_symbol_ir(symbol)
            assert ir_val is not None, f"Identifier '{name}' has no IR mapping"
            return ir_val
    
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
    
    def generic_visit(self, node: ast.ASTNode, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.{type(node).__name__}() not implemented")



if __name__ == "__main__":
    from vcs.cli import cli
    cli()
