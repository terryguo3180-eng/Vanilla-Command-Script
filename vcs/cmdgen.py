from typing import overload

from vcs import cmd
from vcs import ir
from vcs import irgen as irg
from vcs import utils


class CommandGenerator(ir.IRProcessor):
    def __init__(
        self,
        irgen: irg.IRGenerator,
        fixed_precision=1000,
        dump_cmds=False
    ):
        self.namespace = irgen.namespace
        self.irgen = irgen
        self.builder = cmd.DatapackBuilder(self.namespace)
        self.fixed_precision = fixed_precision
        self.dump_cmds = dump_cmds

        self.call_graph: ir.CallGraph
        self.cur_func: ir.Function
        self.frame_path = cmd.StoragePath(self.namespace, "frames")

    def generate(self) -> cmd.Datapack | None:
        module = self.irgen.generate()
        if module is None:
            return None
        self.process(module)
        datapack = self.builder.datapack
        datapack.build_setup()
        
        if self.dump_cmds:
            utils.print_info(str(datapack) + "\n")

        return datapack

    def emit(self, command: cmd.Command):
        self.builder.emit(command)

    def emit_binary_score_op(
        self,
        op: type[cmd.ScoreOperation],
        lhs: cmd.ScoreVar,
        rhs: cmd.ScoreVar | cmd.ImmValue
    ):
        self.builder.emit_binary_score_op(op, lhs, rhs)

    def with_func(self, name: str):
        return self.builder.with_func(name)
    
    def add_func(self, name: str, mcf: cmd.MCFunction | None = None):
        return self.builder.add_func(name, mcf)
    
    def get_func(self, name: str):
        return self.builder.get_func(name)

    def _get_block_mcfname(self, block: ir.BasicBlock):
        return f"{block.func.name}/{block.name}"
        
    def process_Module(self, mod: ir.Module):
        self.call_graph = mod.build_call_graph()

        for func in mod.functions:
            for block in func.blocks:
                self.add_func(self._get_block_mcfname(block))

        for func in mod.functions:
            self.cur_func = func
            self.process(func)
    
    def process_Function(self, func: ir.Function):
        for block in func.blocks:
            with self.with_func(self._get_block_mcfname(block)):
                self.process(block)
        
        return self.get_func(func.name)
    
    def process_BasicBlock(self, block: ir.BasicBlock):
        for inst in block.instructions:
            self.process(inst)
    
    def process_Comment(self, inst: ir.Comment):
        self.emit(cmd.Comment(inst.value))
    
    def _process_int_binary_instr(self, inst: ir.IntBinaryInstr):
        lhs_var = self.process(inst.lhs)
        rhs_var = self.process(inst.rhs)
        target_var = self.process(inst.target)
        self.emit(cmd.ScoreSet(target_var, lhs_var))
        command_type: type[cmd.ScoreOperation] = {
            ir.IAdd: cmd.ScoreAdd,
            ir.ISub: cmd.ScoreSub,
            ir.IMul: cmd.ScoreMul,
            ir.IDiv: cmd.ScoreDiv,
            ir.IMod: cmd.ScoreMod,
        }[type(inst)]
        self.emit_binary_score_op(command_type, target_var, rhs_var)
    
    def _process_fixed_binary_instr(self, inst: ir.FixedBinaryInstr):
        lhs_var = self.process(inst.lhs)
        rhs_var = self.process(inst.rhs)
        target_var = self.process(inst.target)
        self.emit(cmd.ScoreSet(target_var, lhs_var))
        command_type: type[cmd.ScoreOperation] = {
            ir.XAdd: cmd.ScoreAdd,
            ir.XSub: cmd.ScoreSub,
            ir.XMul: cmd.ScoreMul,
            ir.XDiv: cmd.ScoreDiv,
            ir.XMod: cmd.ScoreMod,
        }[type(inst)]

        # For multiplication/division, we need to adjust for fixed-point precision
        if isinstance(inst, ir.XDiv):
            self.emit_binary_score_op(cmd.ScoreMul, target_var, cmd.ImmValue(self.fixed_precision))

        self.emit_binary_score_op(command_type, target_var, rhs_var)

        if isinstance(inst, ir.XMul):
            self.emit_binary_score_op(cmd.ScoreDiv, target_var, cmd.ImmValue(self.fixed_precision))

    def process_IAdd(self, inst: ir.IAdd):
        self._process_int_binary_instr(inst)

    def process_ISub(self, inst: ir.ISub):
        self._process_int_binary_instr(inst)

    def process_IMul(self, inst: ir.IMul):
        self._process_int_binary_instr(inst)

    def process_IDiv(self, inst: ir.IDiv):
        self._process_int_binary_instr(inst)

    def process_IMod(self, inst: ir.IMod):
        self._process_int_binary_instr(inst)

    def process_XAdd(self, inst: ir.XAdd):
        self._process_fixed_binary_instr(inst)
    
    def process_XSub(self, inst: ir.XSub):
        self._process_fixed_binary_instr(inst)
    
    def process_XMul(self, inst: ir.XMul):
        self._process_fixed_binary_instr(inst)
    
    def process_XDiv(self, inst: ir.XDiv):
        self._process_fixed_binary_instr(inst)
    
    def process_XMod(self, inst: ir.XMod):
        self._process_fixed_binary_instr(inst)

    def process_Not(self, inst: ir.Not):
        value_var = self.process(inst.value)
        target_var = self.process(inst.target)

        self.emit(cmd.Execute([
            # execute store result score <target> if score <value> matches 0
            cmd.StoreScore(target_var, cmd.Result()),
            cmd.IfScore(value_var, cmd.ImmValue(0), utils.EqOp()),
        ], None))
    
    def process_And(self, inst: ir.And):
        lhs_var = self.process(inst.lhs)
        rhs_var = self.process(inst.rhs)
        target_var = self.process(inst.target)

        if isinstance(lhs_var, cmd.ScoreVar) and isinstance(rhs_var, cmd.ScoreVar):
            self.emit(cmd.Execute([
                # execute store result score <target>\
                # unless score <lhs> matches 0 unless score <rhs> matches 0
                cmd.StoreScore(target_var, cmd.Result()),
                cmd.IfScore(lhs_var, cmd.ImmValue(0), utils.NeOp()),
                cmd.IfScore(rhs_var, cmd.ImmValue(0), utils.NeOp()),
            ], None))
        elif isinstance(lhs_var, cmd.ImmValue) and isinstance(rhs_var, cmd.ScoreVar):
            self.emit(cmd.Execute([
                # execute store result score <target> unless <rhs> matches 0
                cmd.StoreScore(target_var, cmd.Result()),
                cmd.IfScore(rhs_var, cmd.ImmValue(0), utils.NeOp()),
            ], None))
        elif isinstance(lhs_var, cmd.ScoreVar) and isinstance(rhs_var, cmd.ImmValue):
            self.emit(cmd.Execute([
                # execute store result score <target> unless <lhs> matches 0
                cmd.StoreScore(target_var, cmd.Result()),
                cmd.IfScore(lhs_var, cmd.ImmValue(0), utils.NeOp()),
            ], None))
        else:
            raise NotImplementedError("Unsupported combination of operands for And operation")
    
    def process_Or(self, inst: ir.Or):
        lhs_var = self.process(inst.lhs)
        rhs_var = self.process(inst.rhs)
        target_var = self.process(inst.target)

        if isinstance(lhs_var, cmd.ScoreVar) and isinstance(rhs_var, cmd.ScoreVar):
            self.emit(cmd.Execute([
                # execute store result score <target> unless score <lhs> matches 0
                cmd.StoreScore(target_var, cmd.Result()),
                cmd.IfScore(lhs_var, cmd.ImmValue(0), utils.NeOp()),
            ], None))
            self.emit(cmd.Execute([
                # execute unless score <rhs> matches 0 run scoreboard players set <target> 1
                cmd.IfScore(rhs_var, cmd.ImmValue(0), utils.NeOp()),
            ],
                cmd.ScoreSet(target_var, cmd.ImmValue(1)),
            ))
        elif isinstance(lhs_var, cmd.ImmValue) and isinstance(rhs_var, cmd.ScoreVar):
            self.emit(cmd.Execute([
                # execute unless score <rhs> matches 0 run scoreboard players set <target> 1
                cmd.IfScore(rhs_var, cmd.ImmValue(0), utils.NeOp()),
            ],
                cmd.ScoreSet(target_var, cmd.ImmValue(1)),
            ))
        elif isinstance(lhs_var, cmd.ScoreVar) and isinstance(rhs_var, cmd.ImmValue):
            self.emit(cmd.Execute([
                # execute unless score <lhs> matches 0 run scoreboard players set <target> 1
                cmd.IfScore(lhs_var, cmd.ImmValue(0), utils.NeOp()),
            ],
                cmd.ScoreSet(target_var, cmd.ImmValue(1)),
            ))
        else:
            raise NotImplementedError("Unsupported combination of operands for Or operation")

    def _process_int_comparison(self, inst: ir.ICmp | ir.XCmp):
        lhs_var = self.process(inst.lhs)
        rhs_var = self.process(inst.rhs)
        target_var = self.process(inst.target)
        op = inst.op

        if isinstance(lhs_var, cmd.ImmValue) and isinstance(rhs_var, cmd.ScoreVar):
            # Constant in the left, flip the comparison operator
            op = {
                utils.EqOp: utils.EqOp,
                utils.NeOp: utils.NeOp,
                utils.LtOp: utils.GtOp,
                utils.GtOp: utils.LtOp,
                utils.LeOp: utils.GeOp,
                utils.GeOp: utils.LeOp,
            }[type(op)]()
            lhs_var, rhs_var = rhs_var, lhs_var

        assert isinstance(lhs_var, cmd.ScoreVar)

        self.emit(cmd.Execute([
            # execute store result score <target> if <comparison>
            cmd.StoreScore(target_var, cmd.Result()),
            cmd.IfScore(lhs_var, rhs_var, op),
        ], None))

    def process_ICmp(self, inst: ir.ICmp):
        self._process_int_comparison(inst)
    
    def process_XCmp(self, inst: ir.XCmp):
        self._process_int_comparison(inst)
    
    def process_IntAssign(self, inst: ir.IntAssign):
        target_var = self.process(inst.target)
        value_var = self.process(inst.value)
        self.emit(cmd.ScoreSet(target_var, value_var))
    
    def process_FixedAssign(self, inst: ir.FixedAssign):
        target_var = self.process(inst.target)
        value_var = self.process(inst.value)
        self.emit(cmd.ScoreSet(target_var, value_var))

    def process_IntToFixed(self, inst: ir.IntToFixed):
        target_var = self.process(inst.target)
        value_var = self.process(inst.value)
        self.emit(cmd.ScoreSet(target_var, value_var))
        self.emit_binary_score_op(cmd.ScoreMul, target_var, cmd.ImmValue(self.fixed_precision))

    def process_FixedToInt(self, inst: ir.FixedToInt):
        target_var = self.process(inst.target)
        value_var = self.process(inst.value)
        self.emit(cmd.ScoreSet(target_var, value_var))
        self.emit_binary_score_op(cmd.ScoreDiv, target_var, cmd.ImmValue(self.fixed_precision))

    def process_Call(self, inst: ir.Call):
        callee = self.get_func(self._get_block_mcfname(inst.func.entry_block))
        assert callee is not None

        obj = self._get_objective(inst.func)

        # Store arguments in the function's objective
        for name, value in inst.args.items():
            if ir.int_typed(value) or ir.fixed_typed(value):
                var = cmd.ScoreVar(name, obj)
                self.emit(cmd.ScoreSet(var, self.process(value)))
            else:
                raise NotImplementedError(f"Unsupported argument type: {type(value)}")

        if inst.target is None:  # return value is ignored
            self.emit(cmd.Call(callee))
        elif ir.int_typed(inst.target) or ir.fixed_typed(inst.target):
            var = self.process(inst.target)
            self.emit(cmd.Execute([cmd.StoreScore(var, cmd.Result())], cmd.Call(callee)))
        else:
            raise NotImplementedError(f"Unsupported call target type: {type(inst.target)}")
    
    def process_Push(self, inst: ir.Push):
        self.emit(cmd.StorageAppendValue(self.frame_path, cmd.ImmNBT('{}')))
        cur_frame = self.frame_path[-1]

        # Store the values in the current frame's storage
        for i, value in enumerate(inst.values):
            if ir.int_typed(value) or ir.fixed_typed(value):
                var = self.process(value)
                self.emit(cmd.Execute([
                    cmd.StoreStorage(getattr(cur_frame, f"v{i}"), cmd.Result(), cmd.IntType(), 1)
                ], cmd.ScoreGet(var)))
            else:
                raise NotImplementedError(f"Unsupported value type for Push: {type(value)}")
    
    def process_Pop(self, inst: ir.Pop):
        cur_frame = self.frame_path[-1]

        # Retrieve the values from the current frame's storage and 
        # store them in the target variables
        for i, value in enumerate(inst.values):
            if ir.int_typed(value) or ir.fixed_typed(value):
                var = self.process(value)
                self.emit(cmd.Execute([
                    cmd.StoreScore(var, cmd.Result())
                ], cmd.StorageGet(getattr(cur_frame, f"v{i}"))))
            else:
                raise NotImplementedError(f"Unsupported value type for Pop: {type(value)}")
            
        self.emit(cmd.StorageDel(cur_frame))

    def process_Return(self, inst: ir.Return):
        if inst.value is None:  # return without a value, default to returning 0
            self.emit(cmd.ReturnValue(cmd.ImmValue(0)))

        elif ir.int_typed(inst.value) or ir.fixed_typed(inst.value):
            var = self.process(inst.value)
            if isinstance(var, cmd.ImmValue):
                self.emit(cmd.ReturnValue(var))
            else:
                self.emit(cmd.ReturnRun(cmd.ScoreGet(var)))
        else:
            raise NotImplementedError(f"Unsupported return value type: {type(inst.value)}")

    def process_Goto(self, inst: ir.Goto):
        mcfname = self._get_block_mcfname(inst.label)
        mcf = self.get_func(mcfname)
        assert mcf is not None, f"Function {mcfname} not found for Goto instruction"
        # function <label>
        self.emit(cmd.ReturnRun(cmd.Call(mcf)))
    
    def process_Branch(self, inst: ir.Branch):
        body_mcfname = self._get_block_mcfname(inst.true)
        else_mcfname = self._get_block_mcfname(inst.false)
        cond_var = self.process(inst.cond)

        body_mcf = self.get_func(body_mcfname)
        else_mcf = self.get_func(else_mcfname)

        assert body_mcf is not None, f"Function {body_mcfname} not found for Branch instruction"
        assert else_mcf is not None, f"Function {else_mcfname} not found for Branch instruction"

        self.emit(cmd.Execute([
            # execute unless score <cond> matches 0 run return run function <true>
            cmd.IfScore(cond_var, cmd.ImmValue(0), utils.NeOp()),
        ],
            cmd.ReturnRun(cmd.Call(body_mcf))
        ))
        # function <false>
        self.emit(cmd.ReturnRun(cmd.Call(else_mcf)))

    def process_Tell(self, inst: ir.Tell):
        var = self.process(inst.value)

        if isinstance(var, cmd.ImmValue):
            self.emit(cmd.Tellraw(cmd.Selector('a', {}), str(var.value)))
        else:
            self.emit(cmd.Tellraw(
                cmd.Selector('a', {}),
                {"score": {"name": var.name, "objective": var.objective}}
            ))

    def _get_objective(self, func: ir.Function | None = None):
        if func is None:
            func = self.cur_func
        assert func is not None, "No current function context"
        return f"{self.namespace}.{func.name}"

    def process_NamedValue(self, value: ir.NamedValue):
        objective = self._get_objective()
        if ir.int_typed(value) or ir.fixed_typed(value):
            return cmd.ScoreVar(value.name, objective)
        
        raise NotImplementedError(f"Unsupported NamedValue type: {type(value)}")
    
    def process_Constant(self, value: ir.Constant):
        if ir.int_typed(value):
            return cmd.ImmValue(value.int_value())
        if ir.fixed_typed(value):
            return cmd.ImmValue(value.fixed_value(self.fixed_precision))
        
        raise NotImplementedError(f"Unsupported Constant type: {type(value)}")

    @overload
    def process(self, inst: ir.Module) -> None: ...
    @overload
    def process(self, inst: ir.Instruction) -> None: ...
    @overload
    def process(self, inst: ir.NamedValue[ir.IntType] | ir.NamedValue[ir.FixedType]) -> cmd.ScoreVar: ...
    @overload
    def process(self, inst: ir.Constant[ir.IntType] | ir.Constant[ir.FixedType]) -> cmd.ImmValue: ...
    @overload
    def process(self, inst: ir.Value[ir.IntType] | ir.Value[ir.FixedType]) -> cmd.ScoreVar | cmd.ImmValue: ...
    @overload
    def process(self, inst: ir.Value) -> cmd.CommandValue: ...

    def process(self, inst):
        return super().process(inst)


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
