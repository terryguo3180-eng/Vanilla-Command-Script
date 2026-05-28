from dataclasses import dataclass
from typing import Any


class IRInstr: ...

class Operand: ...

@dataclass
class CommentInstr(IRInstr):
    value: str

    def __repr__(self):
        return self.value

class ImmOperand(Operand):
    value: Any

@dataclass
class IImm(ImmOperand):
    value: int

    def __repr__(self):
        return f"(i)${self.value}"

@dataclass
class BImm(ImmOperand):
    value: bool

    def __repr__(self):
        return f"(b)${self.value}"

@dataclass
class VarOperand(Operand):
    name: str

    def __hash__(self):
        return hash(self.name)

class IVar(VarOperand):
    def __repr__(self):
        return f"(i){self.name}"

class BVar(VarOperand):
    def __repr__(self):
        return f"(b){self.name}"

@dataclass
class ICmpInstr(IRInstr):
    flag: BVar
    target: IVar | IImm
    source: IVar | IImm

class ILt(ICmpInstr):
    def __repr__(self):
        return f"ilt {self.flag} {self.target} {self.source}"
    
class IGt(ICmpInstr):
    def __repr__(self):
        return f"igt {self.flag} {self.target} {self.source}"
    
class ILtE(ICmpInstr):
    def __repr__(self):
        return f"ile {self.flag} {self.target} {self.source}"
    
class IGtE(ICmpInstr):
    def __repr__(self):
        return f"ige {self.flag} {self.target} {self.source}"
    
class IEq(ICmpInstr):
    def __repr__(self):
        return f"ieq {self.flag} {self.target} {self.source}"
    
class INotEq(ICmpInstr):
    def __repr__(self):
        return f"ine {self.flag} {self.target} {self.source}"

@dataclass
class BinInstr[T: VarOperand, U: Operand](IRInstr):
    target: T
    source: U

class IBinInstr(BinInstr[IVar, IImm | IVar]): ...

class IMov(IBinInstr):
    def __repr__(self):
        return f"imov {self.target} {self.source}"
    
class IAdd(IBinInstr):
    def __repr__(self):
        return f"iadd {self.target} {self.source}"
    
class ISub(IBinInstr):
    def __repr__(self):
        return f"isub {self.target} {self.source}"
    
class IMul(IBinInstr):
    def __repr__(self):
        return f"imul {self.target} {self.source}"
    
class IDiv(IBinInstr):
    def __repr__(self):
        return f"idiv {self.target} {self.source}"
    
class IMod(IBinInstr):
    def __repr__(self):
        return f"imod {self.target} {self.source}"
    
class IMin(IBinInstr):
    def __repr__(self):
        return f"imin {self.target} {self.source}"
    
class IMax(IBinInstr):
    def __repr__(self):
        return f"imax {self.target} {self.source}"
    
class ISwp(IBinInstr):
    def __repr__(self):
        return f"iswp {self.target} {self.source}"
    
class BMov(BinInstr[BVar, BImm | BVar]):
    def __repr__(self):
        return f"bmov {self.target} {self.source}"

@dataclass
class INeg(IRInstr):
    var: IVar

    def __repr__(self):
        return f"ineg {self.var}"


@dataclass
class BtoI(IRInstr):
    source: BVar | BImm
    target: IVar

    def __repr__(self):
        return f"btoi {self.source} {self.target}"

@dataclass
class ItoB(IRInstr):
    source: IVar | IImm
    target: BVar

    def __repr__(self):
        return f"itob {self.source} {self.target}"

class BBinInstr(BinInstr[BVar, BImm | BVar | IImm | IVar]): ...

class BAnd(BBinInstr):
    def __repr__(self):
        return f"band {self.target} {self.source}"
    
class BOr(BBinInstr):
    def __repr__(self):
        return f"bor {self.target} {self.source}"

@dataclass
class BNot(IRInstr):
    var: BVar

    def __repr__(self):
        return f"bnot {self.var}"

@dataclass
class LabelVar(Operand):
    name: str
    
    def __repr__(self):
        return f"%{self.name}"

    def __hash__(self):
        return hash(self.name)

@dataclass
class Label(IRInstr):
    var: LabelVar
    
    def __repr__(self):
        return f"{self.var}:"

@dataclass
class Br(IRInstr):
    flag: BVar
    true: LabelVar
    false: LabelVar | None = None

    def __repr__(self):
        if self.false is None:
            return f"br {self.flag} {self.true}"
        return f"br {self.flag} {self.true} {self.false}"

@dataclass
class BrNot(Br):
    def __repr__(self):
        if self.false is None:
            return f"brnot {self.flag} {self.true}"
        return f"brnot {self.flag} {self.true} {self.false}"

@dataclass
class Goto(IRInstr):
    label: LabelVar

    def __repr__(self):
        return f"goto {self.label}"

@dataclass
class Call(IRInstr):
    label: LabelVar

    def __repr__(self):
        return f"call {self.label}"

class Ret(IRInstr):
    def __repr__(self):
        return "ret"

class Push(IRInstr):
    def __repr__(self):
        return "push"

class Pop(IRInstr):
    def __repr__(self):
        return "pop"

@dataclass
class Preserve(IRInstr):
    value: VarOperand

    def __repr__(self):
        return f"preserve {self.value}"

