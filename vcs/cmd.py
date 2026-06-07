from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar

from vcs import utils


class Command: ...


@dataclass
class Comment(Command):
    value: str

    def __str__(self):
        if not self.value.startswith('#'):
            return '# ' + self.value.lstrip()
        return self.value


class CommandValue: ...


@dataclass
class ScoreVar(CommandValue):
    name: str
    objective: str

    def __str__(self):
        return f"{self.name} {self.objective}"


@dataclass
class ImmValue(CommandValue):
    value: int

    def __str__(self):
        return str(self.value)


@dataclass
class ScoreOperation[T: CommandValue](Command):
    target: ScoreVar
    source: T


@dataclass
class SimpleScoreOperation(ScoreOperation):
    target: ScoreVar
    source: ScoreVar | ImmValue

    _imm_op: ClassVar[str]
    _var_op: ClassVar[str]

    def __str__(self):
        if isinstance(self.source, ImmValue):
            return f"scoreboard players {self._imm_op} {self.target} {self.source}"
        return f"scoreboard players operation {self.target} {self._var_op} {self.source}"


@dataclass
class ScoreGet(Command):
    source: ScoreVar
    
    def __str__(self):
        return f"scoreboard players get {self.source}"


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
    target: ScoreVar
    source: ScoreVar

    _var_op: ClassVar[str]

    def __str__(self):
        assert isinstance(self.source, ScoreVar)
        return f"scoreboard players operation {self.target} {self._var_op} {self.source}"


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
class Execute[T: ExecuteClause](Command):
    clauses: list[T]
    run_clause: Command | None

    def __str__(self):
        if not self.clauses:
            assert self.run_clause is not None, "'execute' command cannot be empty"
            return str(self.run_clause)
        
        clauses = self.clauses

        # If the run clause is also an execute command, merge all its clauses
        cmd = self
        while isinstance(cmd.run_clause, Execute):
            clauses.extend(cmd.clauses)
            cmd = cmd.run_clause

        clause_chain = " ".join(str(cond) for cond in clauses)
        if self.run_clause is not None:
            return f"execute {clause_chain} run {self.run_clause}"
        
        return f"execute {clause_chain}"


class ExecuteClause: ...


@dataclass
class StoreScore(ExecuteClause):
    target: ScoreVar
    mode: StoreMode


class StoreMode(metaclass=utils.SingletonMeta): ...

class Result(StoreMode): ...
class Success(StoreMode): ...


@dataclass
class IfScore(ExecuteClause):
    lhs: ScoreVar
    rhs: ScoreVar | ImmValue
    op: utils.CompareOp

    def __str__(self):
        if isinstance(self.rhs, ScoreVar):
            return f"if score {self.lhs} {self.op} {self.rhs}"
        
        value = self.rhs.value
        match self.op:
            case utils.EqOp():
                return f"if score {self.lhs} matches {value}"
            case utils.NeOp():
                return f"unless score {self.lhs} matches {value}"
            case utils.GtOp():
                return f"if score {self.lhs} matches {value + 1}.."
            case utils.LtOp():
                return f"if score {self.lhs} matches ..{value - 1}"
            case utils.GeOp():
                return f"if score {self.lhs} matches {value}.."
            case _:
                assert isinstance(self.op, utils.LeOp)
                return f"if score {self.lhs} matches ..{value}"


@dataclass
class Call(Command):
    func: MCFunction

    def __str__(self):
        return f"function {self.func}"


@dataclass
class ReturnValue(Command):
    value: ImmValue

    def __str__(self):
        return f"return {self.value.value}"


@dataclass
class ReturnRun(Command):
    run_clause: Command

    def __str__(self):
        return f"return run {self.run_clause}"


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
        func_str = f"[{self}]\n"
        for cmd in self.commands:
            func_str += f"  {cmd}\n"
        return func_str.rstrip()


class Datapack:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.mcfunctions: dict[str, MCFunction] = {}
    
    def add_func(self, name: str):
        mcf = MCFunction(self.namespace, name)
        self.mcfunctions[name] = mcf
        return mcf
    
    def get_func(self, name: str):
        return self.mcfunctions.get(name)

    def __str__(self):
        datapack_str = f"namespace {self.namespace}:\n"
        for name, mcf in self.mcfunctions.items():
            datapack_str += f"  {mcf.get_content().replace('\n', '\n  ')}\n"
        return datapack_str.rstrip()


class DatapackBuilder:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.datapack = Datapack(namespace)
        self.cur_func: MCFunction | None = None

        self.consts: set[int] = set()
        self.const_obj = f"{namespace}.const"

    def add_func(self, name: str):
        return self.datapack.add_func(name)

    def get_func(self, name: str):
        return self.datapack.get_func(name)

    @contextmanager
    def with_func(self, name: str):
        prev = self.cur_func
        cur_func = self.get_func(name)
        assert cur_func is not None, f"Function {name!r} not exist"
        self.cur_func = cur_func
        yield cur_func
        self.cur_func = prev
    
    def emit(self, command: Command):
        assert self.cur_func is not None
        self.cur_func.emit(command)

    def emit_binary_score_op(self, op: type[ScoreOperation], lhs: ScoreVar, rhs: ScoreVar | ImmValue):
        if not issubclass(op, BinaryScoreOperation):
            self.emit(op(lhs, rhs))
            return
        
        if isinstance(rhs, ScoreVar):
            self.emit(op(lhs, rhs))
            return

        self.consts.add(rhs.value)
        self.emit(op(lhs, ScoreVar(str(rhs.value), self.const_obj)))
    