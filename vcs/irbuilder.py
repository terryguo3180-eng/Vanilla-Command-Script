from typing import Callable

from vcs.astnodes import *
from vcs.astnodes import ASTNode
from vcs.errors import *
from vcs.ir import *
from vcs.lexer import Lexer, TokenType
from vcs.parser import Parser


class TempVarAllocator:
    def __init__(self):
        # Maps types to sets of allocated temporary variables
        self.allocated: dict[type[VarOperand], set[VarOperand]] = {}
    
    def allocate[T: VarOperand](self, type: type[T]) -> T:
        if type not in self.allocated:
            self.allocated[type] = set()

        # Find the next available temporary variable name
        i = 0
        names = {var.name for var in self.allocated[type]}
        while True:
            name = f"t{i}"
            if name not in names:
                # Create and store the new variable
                var = type(name)
                self.allocated[type].add(var)
                return var
            i += 1

    def free(self, var):
        # Only free if it's a valid temporary variable
        if not isinstance(var, VarOperand):
            return
        if not self.hasvar(var):
            return
        # Remove from the allocated set
        self.allocated[type(var)].discard(var)

    def hasvar(self, var):
        # Check if we have any allocated variables of this type
        return type(var).__name__ in self.allocated


class IRBuilder(ASTNodeVisitor):
    # Maps AST type classes to IR variable classes
    _vartype_mapping: dict[type[Type], type[VarOperand]] = {
        IntType: IVar,
        BoolType: BVar,
    }
    # Maps operand types to move instruction types
    _varmove_mapping: dict[type[VarOperand | ImmOperand], type[BinInstr]] = {
        IVar: IMov,
        BVar: BMov,
        IImm: IMov,
        BImm: BMov,
    }
    # Maps arithmetic operations to integer binary instructions
    _int_binop_instrs: dict[type[ArithmeticOp], type[IBinInstr]] = {
        AddOp: IAdd,
        SubOp: ISub,
        MulOp: IMul,
        DivOp: IDiv,
        ModOp: IMod,
        MinOp: IMin,
        MaxOp: IMax,
    }
    # Maps comparison operations to integer compare instructions
    _int_cmpop_instrs: dict[type[CompareOp], type[ICmpInstr]] = {
        LtOp: ILt,
        GtOp: IGt,
        LtEOp: ILtE,
        GtEOp: IGtE,
        EqOp: IEq,
        NotEqOp: INotEq,
    }
    # Maps boolean binary operations to boolean binary instructions
    _bool_binop_instrs: dict[type[BinaryBoolOp], type[BBinInstr]] = {
        AndOp: BAnd,
        OrOp: BOr,
    }
    _return_ops: dict[type[Type], Operand] = {
        IntType: IVar("iret"),
        BoolType: IVar("bret"),
    }
    
    def __init__(
        self, errors: ErrorCollector,
        get_error_info_on: Callable[[ASTNode | TokenInfo | None], ErrorInfo],
    ):
        self.errors = errors
        self.get_error_info_on = get_error_info_on
        
        self.func_labels: dict[str, LabelVar] = {}
        self.func_signatures: dict[str, list[Parameter]] = {}
        self.func_returns: dict[str, Type | None] = {}

        # Stack of symbol tables (only supports int type for now)
        self.scopes: list[dict[str, VarOperand]] = []
        self.instructions: list[IRInstr] = []
        self.temp_allocator = TempVarAllocator()
        # Stack of (break_label, continue_label) for nested loops
        self.loop_stack: list[tuple[LabelVar, LabelVar]] = []

        self.labels: set[LabelVar] = set()
        self.current_returntype: Type | None = None

    def report(self, error: CompilerError):
        self.errors.add(error)

    def emit(self, instr: IRInstr):
        # Add an instruction to the current basic block
        self.instructions.append(instr)
    
    def pop_instr(self):
        # Remove the last instruction
        self.instructions.pop()

    def push_scope(self):
        # Enter a new lexical scope
        self.scopes.append({})
    
    def pop_scope(self):
        # Exit the current lexical scope
        self.scopes.pop()
    
    def var_lookup(self, name: str) -> VarOperand | None:
        # Search for a variable from innermost to outermost scope
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def add_function(self, name: str, params: list[Parameter], return_type: Type | None):
        self.func_signatures[name] = params
        self.func_returns[name] = return_type

    def get_funclabel(self, name: str):
        return self.func_labels[name]

    def get_signature(self, name: str):
        return self.func_signatures[name]
    
    def get_returntype(self, name: str):
        return self.func_returns[name]

    def add_var(self, name: str, var: VarOperand):
        # Add a variable to the current scope
        self.scopes[-1][name] = var

    def push_loop(self, break_label: LabelVar, continue_label: LabelVar):
        # Push loop labels onto the stack
        self.loop_stack.append((break_label, continue_label))

    def pop_loop(self):
        # Pop loop labels from the stack
        self.loop_stack.pop()
    
    def get_break_continue_labels(self):
        # Get the current innermost loop's labels
        return self.loop_stack[-1]

    def get_unique_label(self, label: str):
        # Generate a unique label name by appending a number if needed
        label_strs = {label.name for label in self.labels}
        if label in label_strs:
            i = 0
            while True:
                newlabel = f"{label}_{i}"
                if newlabel not in self.labels:
                    label = newlabel
                    break
                i += 1
        var = LabelVar(label)
        self.labels.add(var)
        return var

    def visit_Module(self, node: Module):
        for sub in node.body:
            if isinstance(sub, FunctionDeclaration):
                label = self.get_unique_label(sub.name_token.value)
                self.func_labels[sub.name_token.value] = label
                self.add_function(sub.name_token.value, sub.params, sub.return_type)

        # Visit all top-level declarations
        self.push_scope()
        for sub in node.body:
            self.visit(sub)
        self.pop_scope()

    def visit_FunctionDeclaration(self, node: FunctionDeclaration):
        label = self.get_funclabel(node.name_token.value)
        self.current_returntype = node.return_type
        self.emit(Label(label))
        self.push_scope()
        self.visit(node.body)
        self.pop_scope()
        self.emit(Ret())
    
    def visit_Comment(self, node: Comment):
        self.emit(CommentInstr(node.token.value))

    def visit_Block(self, node: Block):
        # Enter a new scope for the block
        self.push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self.pop_scope()
    
    def visit_ExpressionStatement(self, node: ExpressionStatement):
        # Evaluate the expression (for side effects)
        self.visit(node.expression)
    
    def visit_VariableDeclarationStatement(self, node: VariableDeclarationStatement):
        # Get variable name and type
        name = node.name_token.value
        node_type = type(node.type)
        if node_type not in self._vartype_mapping:
            raise NotImplementedError(f"Type {node_type.__name__!r} not supported :(")

        # Check if variable already declared
        var = self.var_lookup(name)
        if var is not None:
            self.report(VariableDeclared(self.get_error_info_on(node), name))
        else:
            # Create and add new variable
            var = self._vartype_mapping[node_type](name)
            self.add_var(name, var)

        # Handle initialization if present
        if node.value is not None:
            value = self.visit(node.value)
            # Check type compatibility
            if isinstance(value, (BImm, BVar)) and isinstance(node.type, (IntType)):
                self.report(UnassignableType(self.get_error_info_on(node.value), "Bool", "Int"))
                value = IVar("error")
            if isinstance(value, (IImm, IVar)) and isinstance(node.type, (BoolType)):
                self.report(UnassignableType(self.get_error_info_on(node.value), "Int", "Bool"))
                value = IVar("error")

            # Emit move instruction
            instr = self._varmove_mapping[type(var)]
            self.emit(instr(var, value))
            self.temp_allocator.free(value)
    
    def visit_AssignStatement(self, node: AssignStatement):
        # Only support simple identifier assignment for now
        if not isinstance(node.target, Identifier):
            raise NotImplementedError("Only supports assignment to identifiers for now :(")
        
        # Look up the target variable
        name = node.target.token.value
        var = self.var_lookup(name)
        if var is None:
            self.report(UndefinedSymbol(self.get_error_info_on(node.target), name))
            target = IVar(name)  # TODO: Use an ErrorVar instead of IVar
        else:
            target = var

        # Evaluate the right-hand side
        value = self.visit(node.value)
        
        # Check type compatibility
        if isinstance(value, (BImm, BVar)) and isinstance(target, (IImm, IVar)):
            self.report(UnassignableType(self.get_error_info_on(node.value), "Bool", "Int"))
            value = IVar("error")
        if isinstance(value, (IImm, IVar)) and isinstance(target, (BImm, BVar)):
            self.report(UnassignableType(self.get_error_info_on(node.value), "Int", "Bool"))
            value = IVar("error")

        # Emit move instruction
        instr = self._varmove_mapping[type(target)]
        self.emit(instr(target, value))
        self.temp_allocator.free(value)
    
    def visit_AugAssignStatement(self, node: AugAssignStatement):
        # Only support simple identifier augmented assignment for now
        if not isinstance(node.target, Identifier):
            raise NotImplementedError("Only supports augmented assignment to identifiers for now :(")
        
        # Look up the target variable
        name = node.target.token.value
        var = self.var_lookup(name)
        if var is None:
            self.report(UndefinedSymbol(self.get_error_info_on(node.target), name))
            target = IVar(name)  # TODO: Use an ErrorVar instead of IVar
        else:
            target = var
        
        # Only integer variables support augmented assignment
        if not isinstance(target, IVar):
            raise NotImplementedError("Only IVar can be used in augmented assignment :(")

        # Evaluate right-hand side and emit operation
        value = self.visit(node.value)
        instr = self._int_binop_instrs[type(node.op)]
        self.emit(instr(target, value))
        self.temp_allocator.free(value)

    def visit_SwapStatement(self, node: SwapStatement):
        # Get left and right identifiers
        left = node.left
        right = node.right
        
        # Only support swapping identifiers for now
        if not (isinstance(left, Identifier) and isinstance(right, Identifier)):
            raise NotImplementedError("Only supports swapping to identifiers for now :(")
        
        # Emit swap instruction
        self.emit(ISwp(self.visit(left), self.visit(right)))

    def visit_PassStatement(self, node: PassStatement):
        pass

    def visit_ReturnStatement(self, node: ReturnStatement):
        if self.current_returntype is None:
            return_op = self._return_ops[IntType]
        else:
            return_op = self._return_ops[type(self.current_returntype)]
        
        value_op = self.visit(node.value)
        if isinstance(return_op, IVar) and isinstance(value_op, (BImm, BVar)):
            self.report(InvalidType(self.get_error_info_on(node.value), "Bool", "Int"))
        if isinstance(return_op, BVar) and isinstance(value_op, (IImm, IVar)):
            self.report(InvalidType(self.get_error_info_on(node.value), "Int", "Bool"))
        
        self.emit(self._varmove_mapping[type(value_op)](return_op, value_op))
        self.emit(Ret())

    def _convert_int_to_bool(self, var: IVar) -> BVar:
        # Convert integer to boolean by creating a temporary boolean variable
        temp = self.temp_allocator.allocate(BVar)
        self.emit(ItoB(var, temp))
        self.temp_allocator.free(var)  # Original variable no longer needed
        return temp

    def visit_IfStatement(self, node: IfStatement):
        # Evaluate the condition
        test_var = self.visit(node.test)

        # Handle compile-time constant condition
        if isinstance(test_var, (IImm, BImm)):
            if test_var.value:
                self.visit(node.body)
            else:
                self.visit(node.orelse)
            return
    
        # Convert integer condition to boolean if needed
        if isinstance(test_var, IVar):
            test_var = self._convert_int_to_bool(test_var)

        # Create labels for branching
        end_label = self.get_unique_label(f"endif_{node.end_lineno}")

        # Generate conditional branch
        if node.orelse is not None:
            else_label = self.get_unique_label(f"else_{node.orelse.lineno}")
            self.emit(BrNot(test_var, else_label))
        else:
            self.emit(BrNot(test_var, end_label))

        # Free the condition variable
        self.temp_allocator.free(test_var)
        
        # Visit then branch
        self.visit(node.body)

        # Handle else branch if present
        if node.orelse is not None:
            self.emit(Goto(end_label))
            self.emit(Label(else_label))  # type: ignore
            self.visit(node.orelse)
        
        # End of if statement
        self.emit(Label(end_label))

    def visit_WhileStatement(self, node: WhileStatement):
        # Create labels for loop
        while_label = self.get_unique_label(f"while_{node.lineno}")
        continue_label = self.get_unique_label(f"continue_{node.lineno}")
        end_label = self.get_unique_label(f"endwhile_{node.end_lineno}")

        # Evaluate condition
        test_var = self.visit(node.test)
        
        # Handle compile-time constant condition
        if isinstance(test_var, (IImm, BImm)):
            if test_var.value:
                self.emit(Label(while_label))
                self.visit(node.body)
                self.emit(Goto(while_label))
            return
        
        # Setup break/continue labels for nested statements
        self.push_loop(end_label, continue_label)

        # Generate loop structure
        self.emit(Label(while_label))
        if isinstance(test_var, IVar):
            test_var = self._convert_int_to_bool(test_var)
        self.emit(BrNot(test_var, end_label))
        self.temp_allocator.free(test_var)

        self.visit(node.body)

        # Continue point - re-evaluate condition
        self.emit(Label(continue_label))
        test_var = self.visit(node.test)
        if isinstance(test_var, IVar):
            test_var = self._convert_int_to_bool(test_var)

        self.emit(Br(test_var, while_label))
        self.emit(Label(end_label))

    def visit_ForStatement(self, node: ForStatement):
        # Enter new scope for loop variables
        self.push_scope()

        # Create labels for loop control
        loop_start_label = self.get_unique_label(f"for_{node.lineno}")
        continue_label = self.get_unique_label(f"continue_{node.lineno}")
        break_label = self.get_unique_label(f"endfor_{node.end_lineno}")

        # Execute initialization statement
        if node.init_stmt is not None:
            self.visit(node.init_stmt)

        # Start of loop
        self.emit(Label(loop_start_label))

        # Check loop condition if present
        if node.test is not None:
            test_var = self.visit(node.test)

            # Handle compile-time constant condition
            if isinstance(test_var, (IImm, BImm)):
                if not test_var.value:
                    self.emit(Goto(break_label))
                    self.emit(Label(break_label))
                    self.pop_scope()
                    return
            else:
                if isinstance(test_var, IVar):
                    test_var = self._convert_int_to_bool(test_var)
                self.emit(BrNot(test_var, break_label))
                self.temp_allocator.free(test_var)

        # Setup loop labels and execute body
        self.push_loop(break_label, continue_label)
        self.visit(node.body)
        
        # Continue point, execute loop end statement
        self.emit(Label(continue_label))
        if node.end_stmt is not None:
            self.visit(node.end_stmt)

        # Jump back to start
        self.emit(Goto(loop_start_label))
        self.emit(Label(break_label))

        # Clean up
        self.pop_loop()
        self.pop_scope()

    def visit_BreakStatement(self, node: BreakStatement):
        # Jump to the break label of the innermost loop
        self.emit(Goto(self.get_break_continue_labels()[0]))

    def visit_ContinueStatement(self, node: ContinueStatement):
        # Jump to the continue label of the innermost loop
        self.emit(Goto(self.get_break_continue_labels()[1]))

    def visit_BinaryExpression(self, node: BinaryExpression):
        # Evaluate left and right operands
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op

        # Optimize compile-time constant expressions
        if isinstance(left, ImmOperand) and isinstance(right, ImmOperand):
            return self._handle_const_binexpr(left, op, right)

        # Handle boolean binary operations
        if isinstance(op, BinaryBoolOp):
            move_instr = self._varmove_mapping[type(left)]
            bool_instr = self._bool_binop_instrs[type(op)]
            temp = self.temp_allocator.allocate(BVar)
            self.emit(move_instr(temp, left))
            self.temp_allocator.free(left)
            self.emit(bool_instr(temp, right))
            self.temp_allocator.free(right)
            return temp

        # Handle comparison operations
        if isinstance(op, CompareOp):
            # Report errors if booleans appear in comparison
            if isinstance(left, (BImm, BVar)):
                self.report(BoolInComparison(self.get_error_info_on(node.left)))
                left = IVar("error")
            if isinstance(right, (BImm, BVar)):
                self.report(BoolInComparison(self.get_error_info_on(node.right)))
                left = IVar("error")

            assert isinstance(left, (IImm, IVar))
            assert isinstance(right, (IImm, IVar))

            # Emit comparison instruction
            comp_instr = self._int_cmpop_instrs[type(op)]
            flag = self.temp_allocator.allocate(BVar)
            self.emit(comp_instr(flag, left, right))
            return flag
        
        # Handle arithmetic operations
        if isinstance(op, ArithmeticOp):
            # Report errors if booleans appear in arithmetic
            if isinstance(left, (BImm, BVar)):
                self.report(BoolInArithmetic(self.get_error_info_on(node.left)))
                left = IVar("error")
            if isinstance(right, (BImm, BVar)):
                self.report(BoolInArithmetic(self.get_error_info_on(node.right)))
                left = IVar("error")

            assert isinstance(left, (IImm, IVar))
            assert isinstance(right, (IImm, IVar))

            # Emit arithmetic instruction
            instr = self._int_binop_instrs[type(op)]
            temp = self.temp_allocator.allocate(IVar)
            self.emit(IMov(temp, left))
            self.temp_allocator.free(left)
            self.emit(instr(temp, right))
            self.temp_allocator.free(right)
            return temp

    def visit_UnaryExpression(self, node: UnaryExpression):
        operand = node.operand
        op = node.op

        if isinstance(operand, ImmOperand):
            return self._handle_const_unaexpr(operand, op)
        
        var = self.visit(operand)

        match op:
            case PosOp():
                if not isinstance(operand, IVar):
                    self.report(InvalidUnaryType(self.get_error_info_on(operand), "+", "Bool"))
                    return IVar("error")
                return var
            case NegOp():
                if not isinstance(operand, IVar):
                    self.report(InvalidUnaryType(self.get_error_info_on(operand), "-", "Bool"))
                    return IVar("error")

                if not self.temp_allocator.hasvar(var):
                    temp = self.temp_allocator.allocate(IVar)
                    self.emit(IMov(temp, var))
                    var = temp
                self.emit(INeg(var))
                return var
            case NotOp():
                if isinstance(operand, IVar):
                    var = self._convert_int_to_bool(var)
                
                if not self.temp_allocator.hasvar(var):
                    temp = self.temp_allocator.allocate(BVar)
                    self.emit(BMov(temp, var))
                    var = temp
                self.emit(BNot(var))
                return var
    
    def _handle_const_binexpr(self, left: ImmOperand, op: BinaryOp, right: ImmOperand):
        # Extract constant values
        lval = left.value
        rval = right.value
        
        # Compute the result at compile time
        match op:
            case AddOp():
                return IImm(lval + rval)
            case SubOp():
                return IImm(lval - rval)
            case MulOp():
                return IImm(lval * rval)
            case DivOp():
                return IImm(lval // rval)
            case ModOp():
                return IImm(lval % rval)
            case MinOp():
                return IImm(min(lval, rval))
            case MaxOp():
                return IImm(max(lval, rval))
            case LtOp():
                return BImm(lval < rval)
            case GtOp():
                return BImm(lval > rval)
            case LtEOp():
                return BImm(lval <= rval)
            case GtEOp():
                return BImm(lval >= rval)
            case EqOp():
                return BImm(lval == rval)
            case NotEqOp():
                return BImm(lval != rval)
            case AndOp():
                return BImm(lval and rval)
            case OrOp():
                return BImm(lval or rval)
            case _:
                raise NotImplementedError(f"Binary operation {type(op).__name__!r} not supported :(")

    def _handle_const_unaexpr(self, operand: ImmOperand, op: UnaryOp):
        match op:
            case PosOp():
                return IImm(+operand.value)
            case NegOp():
                return IImm(-operand.value)
            case NotOp():
                return BImm(not operand.value)

    def _bind_arguments(
        self,
        name: str,
        params: list[Parameter],
        psargs: list[PositionalArgument],
        kwargs: list[KeywordArgument],
        error_info: ErrorInfo,
    ) -> dict[str, Expression]:
        
        bound: dict[str, Expression] = {}
        bound_names: set[str] = set()
        nodefaults = len([p for p in params if p.default is None])

        for i, arg in enumerate(psargs):
            if i >= len(params):
                if nodefaults == len(params):
                    self.report(ExcessPosArgs(error_info, name, len(params), len(psargs)))
                else:
                    self.report(ExcessPosArgsDefault(error_info, name, nodefaults, len(params), len(psargs)))
                break
            
            param = params[i]
            param_name = param.name_token.value
            
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
                if param.name_token.value == kw_name:
                    target_param = param
                    break
            
            if target_param is None:
                unknown_kwargs.append(kw_name)
                continue
            
            bound[kw_name] = kwarg.value
            bound_names.add(kw_name)

        for param in params:
            param_name = param.name_token.value
            if param_name not in bound_names and param.default is not None:
                bound[param_name] = param.default
                bound_names.add(param_name)

        missing_params = []
        for param in params:
            param_name = param.name_token.value
            if param_name not in bound_names and param.default is None:
                missing_params.append(param_name)

        if missing_params:
            self.report(MissingParams(error_info, name, len(missing_params), missing_params))

        if unknown_kwargs:
            self.report(UnexpectedKwarg(error_info, name, unknown_kwargs))

        return bound

    def visit_CallExpression(self, node: CallExpression):
        callee = node.callee

        if not isinstance(callee, Identifier):
            raise NotImplementedError(f"{type(callee).__name__!r} is not supported as callee :(")
        
        name = callee.token.value
        params = self.get_signature(name)
        label = self.get_funclabel(name)
        returntype = self.get_returntype(name)
        args = node.args

        psargs = [arg for arg in args if isinstance(arg, PositionalArgument)]
        kwargs = [arg for arg in args if isinstance(arg, KeywordArgument)]
        error_info = self.get_error_info_on(node)

        bound = self._bind_arguments(name, params, psargs, kwargs, error_info)
        bounded_ops = {}

        for name, value in bound.items():
            bounded_ops[name] = self.visit(value)

        # Push stack for local variables
        self.emit(Push())

        # Preserve all the parameters from the previous stack
        for name, operand in bounded_ops.items():
            if isinstance(operand, IImm):
                param_op = self.temp_allocator.allocate(IVar)
                self.emit(IMov(param_op, operand))
            elif isinstance(operand, BImm):
                param_op = self.temp_allocator.allocate(BVar)
                self.emit(BMov(param_op, operand))
            else:
                param_op = operand

            self.emit(Preserve(param_op))

        # Call the function
        self.emit(Call(label))

        # Pop the stack
        self.emit(Pop())

        if returntype is None:
            return IImm(0)
        
        return self._return_ops[type(returntype)]

    def visit_Constant(self, node: Constant):
        # Get the constant value
        value = node.value.value
        
        # Convert to appropriate IR constant type
        if node.value.type == TokenType.INT:
            return IImm(int(value))
        if value in ["true", "false"]:
            return BImm(True if value == "true" else False)
        
        raise NotImplementedError(f"Constant type {node.value.type.name!r} not supported :(")

    def visit_Identifier(self, node: Identifier):
        # Look up the identifier in the symbol table
        name = node.token.value
        var = self.var_lookup(name)
        if var is None:
            self.report(UndefinedSymbol(self.get_error_info_on(node), name))
            var = IVar(name)  # TODO: Use ErrorVar instead of IVar

        return var

    def generic_visit(self, node: ASTNode, *args, **kwargs):
        # Default handler for unsupported AST nodes
        raise NotImplementedError(f"AST node {type(node).__name__!r} not supported yet :(")


def main():
    import argparse
    import sys
    
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
    ir_builder = IRBuilder(errors, parser.get_error_info_on)
    ir_builder.visit(tree)

    # Report any errors
    if not errors.ok():
        for issue in errors.issues:
            print(dump_error(issue), file=sys.stderr)
        return

    # Output the generated IR instructions
    for instr in ir_builder.instructions:
        if isinstance(instr, Label):
            print(f"{instr}")
        else:
            print(f"  {instr}")

if __name__ == "__main__":
    main()
