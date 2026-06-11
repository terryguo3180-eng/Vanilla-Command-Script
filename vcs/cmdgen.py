from typing import overload

from vcs import cmd
from vcs import ir
from vcs import irgen as irg
from vcs import utils


class CommandGenerator(ir.IRProcessor):
    def __init__(self, irgen: irg.IRGenerator):
        self.namespace = irgen.namespace
        self.irgen = irgen
        self.builder = cmd.DatapackBuilder(self.namespace)

        self.call_graph: ir.CallGraph
        self.cur_func: ir.Function
        self.frame_path = cmd.StoragePath(self.namespace, "frames")

    @overload
    def process(self, inst: ir.Module) -> None: ...
    @overload
    def process(self, inst: ir.Instruction) -> None: ...
    @overload
    def process(self, inst: ir.NamedValue[ir.IntType]) -> cmd.ScoreVar: ...
    @overload
    def process(self, inst: ir.Constant[ir.IntType]) -> cmd.ImmValue: ...
    @overload
    def process(self, inst: ir.Value[ir.IntType]) -> cmd.ScoreVar | cmd.ImmValue: ...
    @overload
    def process(self, inst: ir.Value) -> cmd.CommandValue: ...

    def process(self, inst):
        return super().process(inst)

    def generate(self) -> cmd.Datapack | None:
        module = self.irgen.generate()
        if module is None:
            return None
        self.process(module)
        self.builder.datapack.build_setup()
        return self.builder.datapack

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
            assert False
    
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
            assert False

    def process_ICmp(self, inst: ir.ICmp):
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
    
    def process_IntAssign(self, inst: ir.IntAssign):
        target_var = self.process(inst.target)
        value_var = self.process(inst.value)
        self.emit(cmd.ScoreSet(target_var, value_var))
    
    def process_Call(self, inst: ir.Call):
        callee = self.get_func(self._get_block_mcfname(inst.func.entry_block))
        assert callee is not None

        for name, value in inst.args.items():
            if ir.int_typed(value):
                var = cmd.ScoreVar(name, inst.func.name)
                self.emit(cmd.ScoreSet(var, self.process(value)))
            else:
                assert False

        if inst.target is None:
            self.emit(cmd.Call(callee))
        elif ir.int_typed(inst.target):
            var = self.process(inst.target)
            self.emit(cmd.Execute([cmd.StoreScore(var, cmd.Result())], cmd.Call(callee)))
        else:
            assert False
    
    def process_Push(self, inst: ir.Push):
        self.emit(cmd.StorageAppendValue(
            self.frame_path, cmd.ImmNBT('{}')
        ))
        cur_frame = self.frame_path[-1]
        for i, value in enumerate(inst.values):
            if ir.int_typed(value):
                var = self.process(value)
                self.emit(cmd.Execute([
                    cmd.StoreStorage(
                        getattr(cur_frame, f"v{i}"), cmd.Result(), cmd.IntType(), 1
                    )
                ], cmd.ScoreGet(var)))
            else:
                assert False
    
    def process_Pop(self, inst: ir.Pop):
        cur_frame = self.frame_path[-1]
        for i, value in enumerate(inst.values):
            if ir.int_typed(value):
                var = self.process(value)
                self.emit(cmd.Execute([
                    cmd.StoreScore(var, cmd.Result())
                ], cmd.StorageGet(getattr(cur_frame, f"v{i}"))))
            else:
                assert False
        self.emit(cmd.StorageDel(cur_frame))

    def process_Return(self, inst: ir.Return):
        if inst.value is None:
            self.emit(cmd.ReturnValue(cmd.ImmValue(0)))
            return
        if ir.int_typed(inst.value):
            var = self.process(inst.value)
            if isinstance(var, cmd.ImmValue):
                self.emit(cmd.ReturnValue(var))
            else:
                self.emit(cmd.ReturnRun(cmd.ScoreGet(var)))
            return
        assert False

    def process_Goto(self, inst: ir.Goto):
        mcfname = self._get_block_mcfname(inst.label)
        mcf = self.get_func(mcfname)
        assert mcf is not None
        # function <label>
        self.emit(cmd.ReturnRun(cmd.Call(mcf)))
    
    def process_Branch(self, inst: ir.Branch):
        body_mcfname = self._get_block_mcfname(inst.true)
        else_mcfname = self._get_block_mcfname(inst.false)
        cond_var = self.process(inst.cond)

        body_mcf = self.get_func(body_mcfname)
        else_mcf = self.get_func(else_mcfname)

        assert body_mcf is not None
        assert else_mcf is not None

        self.emit(cmd.Execute([
            # execute unless score <cond> matches 0 run return run function <true>
            cmd.IfScore(cond_var, cmd.ImmValue(0), utils.NeOp()),
        ],
            cmd.ReturnRun(cmd.Call(body_mcf))
        ))
        # function <false>
        self.emit(cmd.ReturnRun(cmd.Call(else_mcf)))

    def process_NamedValue(self, value: ir.NamedValue):
        objective = self.cur_func.name
        if ir.int_typed(value):
            return cmd.ScoreVar(value.name, objective)
        assert False
    
    def process_Constant(self, value: ir.Constant):
        if ir.int_typed(value):
            return cmd.ImmValue(value.value())
        assert False


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
