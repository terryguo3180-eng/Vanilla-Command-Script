from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypeGuard, overload

from vcs import semantic as sem
from vcs import utils


class Type: ...


class SimpleType(Type, metaclass=utils.SingletonMeta): ...


class IntType(SimpleType):
    def __str__(self):
        return f'Int'


class FixedType(SimpleType):
    def __str__(self):
        return f'Fixed'


class FloatType(SimpleType):
    def __str__(self):
        return f'Float'


class VoidType(SimpleType):
    def __str__(self):
        return 'Void'


class LabelType(SimpleType):
    def __str__(self):
        return 'Label'


@dataclass(frozen=True)
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
    
    def __hash__(self):
        return id(self)


class Value[T: Type]:
    def __init__(self, type: T):
        self.type = type


# Python's type system is stupid

@overload
def int_typed[T: Type](value: NamedValue[T]) -> TypeGuard[NamedValue[IntType]]: ...
@overload
def int_typed[T: Type](value: Constant[T]) -> TypeGuard[Constant[IntType]]: ...
@overload
def int_typed[T: Type](value: Value[T]) -> TypeGuard[Value[IntType]]: ...

def int_typed(value: Value) -> bool:
    return isinstance(value.type, IntType)

@overload
def float_typed[T: Type](value: NamedValue[T]) -> TypeGuard[NamedValue[FloatType]]: ...
@overload
def float_typed[T: Type](value: Constant[T]) -> TypeGuard[Constant[FloatType]]: ...
@overload
def float_typed[T: Type](value: Value[T]) -> TypeGuard[Value[FloatType]]: ...

def float_typed(value: Value) -> bool:
    return isinstance(value.type, FloatType)

@overload
def fixed_typed[T: Type](value: NamedValue[T]) -> TypeGuard[NamedValue[FixedType]]: ...
@overload
def fixed_typed[T: Type](value: Constant[T]) -> TypeGuard[Constant[FixedType]]: ...
@overload
def fixed_typed[T: Type](value: Value[T]) -> TypeGuard[Value[FixedType]]: ...

def fixed_typed(value: Value) -> bool:
    return isinstance(value.type, FixedType)


class NamedValue[T: Type](Value[T]):
    def __init__(self, name: str, type: T):
        super().__init__(type)
        self.name = name
    
    def __str__(self):
        return f"({self.name}: {self.type})"


class Constant[T: Type](Value[T]):
    def __init__(self, value: int | float, type: T):
        super().__init__(type)
        self._value = value

    def int_value(self: Constant[IntType]) -> int:
        assert isinstance(self._value, int)
        return self._value
    
    def fixed_value(self: Constant[FixedType], precision: int) -> int:
        assert isinstance(self._value, float)
        return int(round(self._value * precision))
    
    def float_value(self: Constant[FloatType]) -> float:
        assert isinstance(self._value, float)
        return self._value

    def __str__(self):
        if isinstance(self.type, (IntType, FloatType, FixedType)):
            return str(self._value)
        return super().__str__()


class Instruction: ...


@dataclass(frozen=True)
class Comment(Instruction):
    value: str

    def __str__(self):
        if not self.value.startswith('#'):
            return '# ' + self.value.lstrip()
        return self.value


@dataclass(frozen=True)
class IntBinaryInstr(Instruction):
    lhs: Value[IntType]
    rhs: Value[IntType]
    target: NamedValue[IntType]

    _opname: ClassVar[str]
        
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


@dataclass(frozen=True)
class FloatBinaryInstr(Instruction):
    lhs: Value[FloatType]
    rhs: Value[FloatType]
    target: NamedValue[FloatType]

    _opname: ClassVar[str]
        
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


@dataclass(frozen=True)
class FixedBinaryInstr(Instruction):
    lhs: Value[FixedType]
    rhs: Value[FixedType]
    target: NamedValue[FixedType]

    _opname: ClassVar[str]
        
    def __str__(self):
        return f"{self.target} = {self._opname} {self.lhs}, {self.rhs}"


class XAdd(FixedBinaryInstr):
    _opname = "xadd"


class XSub(FixedBinaryInstr):
    _opname = "xsub"


class XMul(FixedBinaryInstr):
    _opname = "xmul"


class XDiv(FixedBinaryInstr):
    _opname = "xdiv"


class XMod(FixedBinaryInstr):
    _opname = "xmod"


@dataclass(frozen=True)
class IntToFloat(Instruction):
    value: NamedValue[IntType]
    target: NamedValue[FloatType]

    def __str__(self):
        return f"{self.target} = itof {self.value}"
    

@dataclass(frozen=True)
class FloatToInt(Instruction):
    value: NamedValue[FloatType]
    target: NamedValue[IntType]

    def __str__(self):
        return f"{self.target} = ftoi {self.value}"


@dataclass(frozen=True)
class IntToFixed(Instruction):
    value: NamedValue[IntType]
    target: NamedValue[FixedType]

    def __str__(self):
        return f"{self.target} = itox {self.value}"


@dataclass(frozen=True)
class FixedToInt(Instruction):
    value: NamedValue[FixedType]
    target: NamedValue[IntType]

    def __str__(self):
        return f"{self.target} = xtoi {self.value}"


@dataclass(frozen=True)
class FloatToFixed(Instruction):
    value: NamedValue[FloatType]
    target: NamedValue[FixedType]

    def __str__(self):
        return f"{self.target} = ftox {self.value}"


@dataclass(frozen=True)
class FixedToFloat(Instruction):
    value: NamedValue[FixedType]
    target: NamedValue[FloatType]

    def __str__(self):
        return f"{self.target} = xtof {self.value}"


@dataclass(frozen=True)
class Not(Instruction):
    value: NamedValue[IntType]
    target: NamedValue[IntType]

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


@dataclass(frozen=True)
class ICmp(Instruction):
    lhs: Value[IntType]
    rhs: Value[IntType]
    target: NamedValue[IntType]
    op: utils.CompareOp

    def __str__(self):
        return f"{self.target} = icmp {self.op} {self.lhs}, {self.rhs}"


@dataclass(frozen=True)
class XCmp(Instruction):
    lhs: Value[FixedType]
    rhs: Value[FixedType]
    target: NamedValue[IntType]
    op: utils.CompareOp

    def __str__(self):
        return f"{self.target} = xcmp {self.op} {self.lhs}, {self.rhs}"


@dataclass(frozen=True)
class FCmp(Instruction):
    lhs: Value[FloatType]
    rhs: Value[FloatType]
    target: NamedValue[IntType]
    op: utils.CompareOp

    def __str__(self):
        return f"{self.target} = fcmp {self.op} {self.lhs}, {self.rhs}"


@dataclass(frozen=True)
class IntAssign(Instruction):
    value: Value[IntType]
    target: NamedValue[IntType]

    def __str__(self):
        return f"{self.target} = int {self.value}"


@dataclass(frozen=True)
class FloatAssign(Instruction):
    value: Value[FloatType]
    target: NamedValue[FloatType]
    
    def __str__(self):
        return f"{self.target} = float {self.value}"


@dataclass(frozen=True)
class FixedAssign(Instruction):
    value: Value[FixedType]
    target: NamedValue[FixedType]
    
    def __str__(self):
        return f"{self.target} = fixed {self.value}"


@dataclass(frozen=True)
class Return(Instruction):
    value: Value | None = None
    
    def __str__(self):
        if self.value:
            return f"ret {self.value}"
        return "ret void"


@dataclass
class Push(Instruction):
    values: list[NamedValue]

    def __str__(self):
        return f"push {', '.join(str(val) for val in self.values)}"

    def __hash__(self):
        return id(self)


@dataclass
class Pop(Instruction):
    values: list[NamedValue]

    def __str__(self):
        return f"pop {', '.join(str(val) for val in self.values)}"

    def __hash__(self):
        return id(self)


@dataclass
class Call(Instruction):
    func: Function
    args: dict[str, Value]
    target: NamedValue | None = None
    
    def __str__(self):
        args_str = ', '.join(f"{name}={arg}" for name, arg in self.args.items())
        if self.target:
            return f"{self.target} = call {self.func}({args_str})"
        else:
            return f"call {self.func}({args_str})"

    def __hash__(self):
        return id(self)
        

class BasicBlock(NamedValue):
    def __init__(self, name: str, func: Function):
        self.name = name
        self.func = func
        self.type = LabelType()
        self.instructions: list[Instruction] = []
    
    def emit(self, inst: Instruction):
        self.instructions.append(inst)
    
    def insert(self, index: int, inst: Instruction):
        self.instructions.insert(index, inst)

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
    
    def __hash__(self):
        return id(self)


@dataclass
class Branch(Instruction):
    cond: NamedValue[IntType]
    true: BasicBlock
    false: BasicBlock
    
    def __str__(self):
        return f"branch {self.cond} {self.true} {self.false}"
    
    def __hash__(self):
        return id(self)


@dataclass
class Tell(Instruction):
    value: Value[IntType]

    def __str__(self):
        return f"tell {self.value}"
    
    def __hash__(self):
        return id(self)



class Function(NamedValue):
    type: FunctionType

    def __init__(self, name: str, type: FunctionType, param_names: list[str] | None = None):
        Value.__init__(self, type)
        self.name = name
        self.blocks: list[BasicBlock] = []
        self.entry_block = BasicBlock("_", self)
        self.blocks.append(self.entry_block)
        self.param_names = param_names or [f"p{i}" for i in range(len(type.param_types))]

    def create_block(self, name: str) -> BasicBlock:
        block = BasicBlock(name, self)
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

    def build_call_graph(self) -> CallGraph:
        call_graph = CallGraph()
        for func in self.functions:
            call_graph.add_func(func)
        for func in self.functions:
            for block in func.blocks:
                for inst in block.instructions:
                    if isinstance(inst, Call):
                        call_graph.add_call(func, inst.func)
        return call_graph


class CallGraph:
    def __init__(self):
        self.func_calls: dict[Function, list[Function]] = {}
    
    def add_func(self, func: Function):
        self.func_calls[func] = []

    def add_call(self, func: Function, callee: Function):
        if callee not in self.func_calls[func]:
            self.func_calls[func].append(callee)
    
    def is_recursive(self, func: Function) -> bool:
        visited = set()
        
        def _check(func: Function, init: Function) -> bool:
            if func in visited:
                return False
            visited.add(func)
            
            for callee in self.func_calls[func]:
                if callee is init or _check(callee, init):
                    return True
            return False
        
        return _check(func, func)


class IRBuilder:
    def __init__(self):
        self._value_counter = 0
        self._assigned: list[NamedValue] = []
    
    def new_temp[T: Type](self, type: T) -> NamedValue[T]:
        temp = NamedValue[T](f"#t{self._value_counter}", type)
        self._value_counter += 1
        return temp
    
    def reset_value_counter(self):
        self._value_counter = 0

    def set_block(self, block: BasicBlock):
        self.cur_block = block
    
    def emit(self, inst: Instruction):
        self.cur_block.emit(inst)
    
    def insert(self, index: int, inst: Instruction):
        self.cur_block.insert(index, inst)

    def is_terminated(self) -> bool:
        if not self.cur_block.instructions:
            return False
        last = self.cur_block.instructions[-1]
        return isinstance(last, (Return, Branch, Goto))

    def get_irtype(self, semtype: sem.TypeInfo) -> Type:
        if isinstance(semtype, (sem.IntTypeInfo, sem.BoolTypeInfo)):
            return IntType()  # Booleans are represented as integers (0/1)
        if isinstance(semtype, sem.FixedTypeInfo):
            return FixedType()
        if isinstance(semtype, sem.FloatTypeInfo):
            return FloatType()
        if isinstance(semtype, sem.VoidTypeInfo):
            return VoidType()
        raise ValueError(f"Unsupported semantic type: {semtype}")
    

class IRProcessor:
    def process(self, inst: Module | Instruction | Value):
        method = "process_" + type(inst).__name__
        processor = getattr(self, method, self.generic_process)
        return processor(inst)
    
    def generic_process(self, inst: Module | Instruction | Value):
        raise NotImplementedError(f"{type(self).__name__}.{type(inst).__name__}() not implemented")
