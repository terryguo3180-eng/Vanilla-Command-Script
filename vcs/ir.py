from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vcs import utils


class Type: ...


class IntType(Type):
    def __str__(self):
        return f'Int'


class FloatType(Type):
    def __str__(self):
        return f'Float'


class VoidType(Type):
    def __str__(self):
        return 'Void'


class LabelType(Type):
    def __str__(self):
        return 'Label'


@dataclass
class PointerType(Type):
    pointee: Type
    
    def __str__(self):
        return f'{self.pointee}*'


@dataclass
class FunctionType(Type):
    return_type: Type
    param_types: list[Type]
    
    def __str__(self):
        params = ', '.join(str(t) for t in self.param_types)
        return f'({params}) -> {self.return_type}'


class Value:
    def __init__(self, type: Type):
        self.type = type


class NamedValue(Value):
    _counter = 0
    
    def __init__(self, name: str, type: Type):
        super().__init__(type)
        if name == "":
            self.name = f"%{NamedValue._counter}"
            NamedValue._counter += 1
        else:
            self.name = f"%{name}"
    
    def __str__(self):
        return f"({self.name}: {self.type})"


class Constant(Value):
    def __init__(self, value, type: Type):
        super().__init__(type)
        self.value = value
    
    def __str__(self):
        if isinstance(self.type, (IntType, FloatType)):
            return str(self.value)
        return super().__str__()


class Instruction: ...


@dataclass
class Comment(Instruction):
    value: str

    def __str__(self):
        if not self.value.startswith('#'):
            return '# ' + self.value.lstrip()
        return self.value


@dataclass
class IntBinaryInstr(Instruction):
    lhs: Value
    rhs: Value
    target: NamedValue

    _opname: ClassVar[str]

    def __post_init__(self):
        if not (
            isinstance(self.lhs.type, IntType)
            and isinstance(self.rhs.type, IntType)
            and isinstance(self.target.type, IntType)
        ):
            raise ValueError(self)
        
    def __str__(self):
        return f"{self.target} = {self._opname} {self.lhs}, {self.rhs}"


class IAdd(IntBinaryInstr):
    _opname = "iadd"


class ISub(IntBinaryInstr):
    _opname = "isub"


class IMul(IntBinaryInstr):
    _opname = "imul"


class IDiv(IntBinaryInstr):
    _opname = "idiv"


class IMod(IntBinaryInstr):
    _opname = "imod"


@dataclass
class FloatBinaryInstr(Instruction):
    lhs: Value
    rhs: Value
    target: NamedValue

    _opname: ClassVar[str]

    def __post_init__(self):
        if not (
            isinstance(self.lhs.type, FloatType)
            and isinstance(self.rhs.type, FloatType)
            and isinstance(self.target.type, FloatType)
        ):
            raise ValueError(self)
        
    def __str__(self):
        return f"{self.target} = {self._opname} {self.lhs}, {self.rhs}"


class FAdd(FloatBinaryInstr):
    _opname = "fadd"


class FSub(FloatBinaryInstr):
    _opname = "fsub"


class FMul(FloatBinaryInstr):
    _opname = "fmul"


class FDiv(FloatBinaryInstr):
    _opname = "fdiv"


class FMod(FloatBinaryInstr):
    _opname = "fmod"


@dataclass
class IntToFloat(Instruction):
    value: Value
    target: NamedValue

    def __str__(self):
        return f"{self.target} = itof {self.value}"
    

@dataclass
class FloatToInt(Instruction):
    value: Value
    target: NamedValue

    def __str__(self):
        return f"{self.target} = ftoi {self.value}"


@dataclass
class Not(Instruction):
    value: Value
    target: NamedValue

    def __post_init__(self):
        if not (
            isinstance(self.value.type, IntType)
            and isinstance(self.target.type, IntType)
        ):
            raise ValueError(self)

    def __str__(self):
        return f"{self.target} = not {self.value}"


class And(IntBinaryInstr):
    def __str__(self):
        return f"{self.target} = and {self.lhs}, {self.rhs}"


class Or(IntBinaryInstr):
    def __str__(self):
        return f"{self.target} = or {self.lhs}, {self.rhs}"


@dataclass
class ICmp(Instruction):
    left: Value
    right: Value
    target: NamedValue
    op: utils.CompareOp

    def __post_init__(self):
        if not (
            isinstance(self.left.type, IntType)
            and isinstance(self.right.type, IntType)
            and isinstance(self.target.type, IntType)
        ):
            raise ValueError(self)

    def __str__(self):
        return f"{self.target} = icmp {self.op} {self.left}, {self.right}"


@dataclass
class FCmp(Instruction):
    left: Value
    right: Value
    target: NamedValue
    op: utils.CompareOp

    def __post_init__(self):
        if not (
            isinstance(self.left.type, FloatType)
            and isinstance(self.right.type, FloatType)
            and isinstance(self.target.type, IntType)
        ):
            raise ValueError(self)

    def __str__(self):
        return f"{self.target} = fcmp {self.op} {self.left}, {self.right}"


@dataclass
class Load(Instruction):
    value: Value
    target: NamedValue
    
    def __str__(self):
        return f"{self.target} = load {self.value}"


@dataclass
class Return(Instruction):
    value: Value | None = None
    
    def __str__(self):
        if self.value:
            return f"ret {self.value}"
        return f"ret void"


@dataclass
class Call(Instruction):
    func: Function
    args: list[Value]
    target: NamedValue | None = None
    
    def __str__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        if self.target:
            return f"{self.target} = call {self.func}({args_str})"
        else:
            return f"call {self.func}({args_str})"
        

class BasicBlock(NamedValue):
    def __init__(self, name: str):
        self.name = name
        self.type = LabelType()
        self.instructions: list[Instruction] = []
    
    def emit(self, inst: Instruction):
        self.instructions.append(inst)
    
    def get_content(self):
        block_str = f"{self.name}:"
        for inst in self.instructions:
            block_str += f"\n  {inst}"
        return block_str


@dataclass
class Goto(Instruction):
    label: BasicBlock

    def __post_init__(self):
        if not isinstance(self.label, BasicBlock):
            raise ValueError(self)
    
    def __str__(self):
        return f"goto {self.label}"


@dataclass
class Branch(Instruction):
    cond: NamedValue
    true: BasicBlock
    false: BasicBlock

    def __post_init__(self):
        if not (
            isinstance(self.cond.type, IntType)
            and isinstance(self.true, BasicBlock)
            and isinstance(self.false, BasicBlock)
        ):
            raise ValueError(self)
    
    def __str__(self):
        return f"branch {self.cond} {self.true} {self.false}"


class Function(NamedValue):
    type: FunctionType

    def __init__(self, name: str, type: FunctionType, param_names: list[str] | None = None):
        Value.__init__(self, type)
        self.name = name
        self.blocks: list[BasicBlock] = []
        self.entry_block = BasicBlock("entry")
        self.blocks.append(self.entry_block)
        self.param_names = param_names or [f"p{i}" for i in range(len(type.param_types))]

    def create_block(self, name: str) -> BasicBlock:
        block = BasicBlock(name)
        self.blocks.append(block)
        return block

    def get_content(self):
        params = ', '.join(
            f'{n}: {t}' for n, t in zip(self.param_names, self.type.param_types)
        )
        func_str = f"{self.name}({params}) -> {self.type.return_type}:\n"
        for block in self.blocks:
            func_str += f"  {block.get_content().replace('\n', '\n  ')}\n"
        return func_str.rstrip()


class Module:
    def __init__(self, name: str):
        self.name = name
        self.functions: list[Function] = []
    
    def add_func(self, func: Function):
        self.functions.append(func)
    
    def __str__(self):
        module_str = f"module {self.name}:\n"
        for func in self.functions:
            module_str += f"  {func.get_content().replace('\n', '\n  ')}\n"
        return module_str.rstrip()


class IRBuilder:
    def __init__(self):
        self._value_counter = 0
        self._assigned: list[NamedValue] = []
    
    def new_temp(self, type: Type):
        temp = NamedValue(f"t{self._value_counter}", type)
        self._value_counter += 1
        return temp
    
    def set_block(self, block: BasicBlock):
        self.cur_block = block
    
    def emit(self, inst: Instruction):
        self.cur_block.emit(inst)
    
    def call(self, func: Function, args: list[Value], target: NamedValue | None):
        if not isinstance(func.type.return_type, VoidType):
            self.emit(Call(func, args, target))
        else:
            self.emit(Call(func, args))
    