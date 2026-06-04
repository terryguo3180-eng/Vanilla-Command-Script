from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class ScoreOperation(Command):
    var: ScoreVar
    value: ScoreVar | ImmValue


class ScoreSet(ScoreOperation):
    def __str__(self):
        if isinstance(self.value, ImmValue):
            return f"scoreboard players set {self.var} {self.value}"
        return f"scoreboard players operation {self.var} = {self.value}"


class ScoreAdd(ScoreOperation):
    def __str__(self):
        if isinstance(self.value, ImmValue):
            return f"scoreboard players add {self.var} {self.value}"
        return f"scoreboard players operation {self.var} += {self.value}"


class ScoreSub(ScoreOperation):
    def __str__(self):
        if isinstance(self.value, ImmValue):
            return f"scoreboard players remove {self.var} {self.value}"
        return f"scoreboard players operation {self.var} += {self.value}"


class ScoreMul(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} *= {self.value}"


class ScoreDiv(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} /= {self.value}"


class ScoreMod(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} %= {self.value}"


class ScoreMin(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} < {self.value}"


class ScoreMax(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} > {self.value}"


class ScoreSwp(ScoreOperation):
    def __str__(self):
        assert isinstance(self.value, ScoreVar)
        return f"scoreboard players operation {self.var} >< {self.value}"


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
            case utils.NotEqOp():
                return f"unless score {self.left} matches {value}"
            case utils.GtOp():
                return f"if score {self.left} matches {value + 1}.."
            case utils.LtOp():
                return f"if score {self.left} matches ..{value - 1}"
            case utils.GtEOp():
                return f"if score {self.left} matches {value}.."
            case _:
                assert isinstance(self.op, utils.LtEOp)
                return f"if score {self.left} matches ..{value}"
