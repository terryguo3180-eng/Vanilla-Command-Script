from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar

from vcs import utils


class Command: ...


@dataclass
class ScoreVar:
    name: str
    objective: str

    def __str__(self):
        return f"{self.name} {self.objective}"


@dataclass
class ImmValue:
    value: int

    def __str__(self):
        return str(self.value)


class ScoreOperation(Command): ...


@dataclass
class SimpleScoreOperation(ScoreOperation):
    var: ScoreVar
    value: ScoreVar | ImmValue

    _imm_op: ClassVar[str]
    _var_op: ClassVar[str]

    def __str__(self):
        if isinstance(self.value, ImmValue):
            return f"scoreboard players {self._imm_op} {self.var} {self.value}"
        return f"scoreboard players operation {self.var} {self._var_op} {self.value}"


class ScoreSet(SimpleScoreOperation):
    _imm_op = "set"
    _var_op = "="


class ScoreAdd(SimpleScoreOperation):
    _imm_op = "add"
    _var_op = "+="


class ScoreSub(SimpleScoreOperation):
    _imm_op = "remove"
    _var_op = "-="


@dataclass
class BinaryScoreOperation(ScoreOperation):
    var: ScoreVar
    value: ScoreVar

    _var_op: ClassVar[str]

    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} {self._var_op} {self.value}"


class ScoreMul(BinaryScoreOperation):
    _var_op = "*="


class ScoreDiv(BinaryScoreOperation):
    _var_op = "/="


class ScoreMod(BinaryScoreOperation):
    _var_op = "%="


class ScoreMin(BinaryScoreOperation):
    _var_op = "<"


class ScoreMax(BinaryScoreOperation):
    _var_op = ">"


class ScoreSwp(BinaryScoreOperation):
    _var_op = "><"


@dataclass
class Execute(Command):
    conds: list[ExecuteCond]
    run_cmd: Command

    def __str__(self):
        if not self.conds:
            return str(self.run_cmd)
        cond_chain = " ".join(str(cond) for cond in self.conds)
        return f"execute {cond_chain} run {self.run_cmd}"


class ExecuteCond: ...


class IfScore(ExecuteCond):
    left: ScoreVar
    right: ScoreVar | ImmValue
    op: utils.CompareOp

    def __str__(self):
        if isinstance(self.right, ScoreVar):
            return f"if score {self.left} {self.op} {self.right}"
        
        value = self.right.value
        match self.op:
            case utils.EqOp():
                return f"if score {self.left} matches {value}"
            case utils.NeOp():
                return f"unless score {self.left} matches {value}"
            case utils.GtOp():
                return f"if score {self.left} matches {value + 1}.."
            case utils.LtOp():
                return f"if score {self.left} matches ..{value - 1}"
            case utils.GeOp():
                return f"if score {self.left} matches {value}.."
            case _:
                assert isinstance(self.op, utils.LeOp)
                return f"if score {self.left} matches ..{value}"


class Call(Command):
    func: MCFunction

    def __str__(self):
        return f"function {self.func}"


class MCFunction:
    def __init__(self, namespace: str, name: str):
        self.namespace = namespace
        self.name = name
        self.commands: list[Command] = []
    
    def emit(self, command: Command):
        self.commands.append(command)

    def __str__(self):
        return f"{self.namespace}:{self.name}"
    
    def get_content(self):
        func_str = f"[{self}]"
        for cmd in self.commands:
            func_str += f"  {cmd}\n"
        return func_str.rstrip()


class Datapack:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.mcfunctions: list[MCFunction] = []
    
    def add_func(self, name: str):
        mcf = MCFunction(self.namespace, name)
        self.mcfunctions.append(mcf)
        return mcf


class DatapackBuilder:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.datapack = Datapack(namespace)
        self.cur_func: MCFunction

    @contextmanager
    def with_func(self, name: str):
        prev = self.cur_func
        cur_func = self.datapack.add_func(name)
        self.cur_func = cur_func
        yield cur_func
        self.cur_func = prev
    
    def emit(self, command: Command):
        self.cur_func.emit(command)
