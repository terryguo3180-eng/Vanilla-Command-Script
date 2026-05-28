import operator
from typing import Callable, cast

from vcs.errors import *
from vcs.ir import *


class CommandBuilder:
    _scoreboard_ops: dict[type[IBinInstr], dict[str, str]] = {
        IMov: {"imm": "set", "var": "="},
        IAdd: {"imm": "add", "var": "+="},
        ISub: {"imm": "remove", "var": "-="},
        IMul: {"var": "*="},
        IDiv: {"var": "/="},
        IMod: {"var": "%="},
        ISwp: {"var": "><"},
        IMin: {"var": "<"},
        IMax: {"var": ">"},
    }
    _scoreboard_cmp: dict[type[ICmpInstr], dict[str, str | Callable]] = {
        ILt: {
            "iv": lambda v: f"{v + 1}..",
            "vi": lambda v: f"..{v - 1}",
            "op": operator.lt,
            "opstr": "<",
            "cond": "if"
        },
        IGt: {
            "iv": lambda v: f"..{v - 1}",
            "vi": lambda v: f"{v + 1}..",
            "op": operator.gt,
            "opstr": ">",
            "cond": "if"
        },
        ILtE: {
            "iv": lambda v: f"{v}..",
            "vi": lambda v: f"..{v}",
            "op": operator.le,
            "opstr": "<=",
            "cond": "if"
        },
        IGtE: {
            "iv": lambda v: f"..{v}",
            "vi": lambda v: f"{v}..",
            "op": operator.gt,
            "opstr": ">=",
            "cond": "if"
        },
        IEq: {
            "iv": lambda v: v,
            "vi": lambda v: v,
            "op": operator.eq,
            "opstr": "=",
            "cond": "if"
        },
        INotEq: {
            "iv": lambda v: v,
            "vi": lambda v: v,
            "op": operator.ne,
            "opstr": "=",
            "cond": "unless"
        },
    }

    def __init__(self, namespace: str, instructions: list[IRInstr]):
        self.namespace = namespace
        self.instructions = instructions
        self.mcfunctions: dict[str, list[str]] = {"main": []}
        self.mcfstack: list[str] = ["main"]
        self.constants: set[int] = set()

        self.emit(f"scoreboard objectives add {namespace} dummy")

    def open_func(self, func: str):
        self.mcfstack.append(func)
        self.mcfunctions[func] = []

    def close_func(self):
        self.mcfstack.pop()

    def emit(self, cmd):
        self.mcfunctions[self.mcfstack[-1]].append(cmd)

    def build(self):
        for instr in self.instructions:
            getattr(self, f"build_{type(instr).__name__}")(instr)
        for const in sorted(self.constants, reverse=True):
            self.mcfunctions["main"].insert(1, f"scoreboard players set {const} {self.namespace} {const}")

    def build_CommentInstr(self, instr: CommentInstr):
        assert instr.value.startswith("#")
        self.emit(instr.value)

    def _build_binary_op(self, instr: IBinInstr):
        op_type = type(instr)

        if not isinstance(instr.target, IVar):
            raise NotImplementedError(f"Unsupported target operand for {op_type}")
        
        obj = self.namespace
        target = instr.target.name
        ops = self._scoreboard_ops[op_type]
        
        if isinstance(instr.source, IImm):
            if op_type == "ISwp":
                raise ValueError("Cannot use IImm object for ISwp instruction")
            
            if "imm" not in ops:
                self.constants.add(instr.source.value)
                self.emit(f"scoreboard players operation {target} {obj} {ops["var"]} {instr.source.value} {obj}")
            else:
                self.emit(f"scoreboard players {ops["imm"]} {target} {obj} {instr.source.value}")
        elif isinstance(instr.source, IVar):
            self.emit(f"scoreboard players operation {target} {obj} {ops["var"]} {instr.source.name} {obj}")
    
    def build_IAdd(self, instr): self._build_binary_op(instr)
    def build_ISub(self, instr): self._build_binary_op(instr)
    def build_IMul(self, instr): self._build_binary_op(instr)
    def build_IDiv(self, instr): self._build_binary_op(instr)
    def build_IMod(self, instr): self._build_binary_op(instr)
    def build_IMov(self, instr): self._build_binary_op(instr)
    def build_IMin(self, instr): self._build_binary_op(instr)
    def build_IMax(self, instr): self._build_binary_op(instr)
    def build_ISwp(self, instr): self._build_binary_op(instr)

    def _build_comp_op(self, instr: ICmpInstr):
        op_type = type(instr)
        forms = self._scoreboard_cmp[op_type]

        obj = self.namespace
        flag = instr.flag.name

        target = instr.target
        source = instr.source

        prefix = f"execute store result score {flag} {obj}"

        iv = cast(Callable, forms["iv"])
        vi = cast(Callable, forms["vi"])
        op = cast(Callable, forms["op"])
        opstr = cast(str, forms["opstr"])

        if isinstance(target, IImm) and isinstance(source, IVar):
            self.emit(f"{prefix} if score {source.name} {obj} matches {iv(target.value)}")
        elif isinstance(target, IVar) and isinstance(source, IImm):
            self.emit(f"{prefix} if score {target.name} {obj} matches {vi(source.value)}")
        elif isinstance(target, IImm) and isinstance(source, IImm):
            self.emit(f"scoreboard players set {flag} {obj} {int(op(target.value, source.value))}")
        elif isinstance(target, IVar) and isinstance(source, IVar):
            self.emit(f"{prefix} if score {target.name} {obj} {opstr} {source.name} {obj}")
        else:
            raise NotImplementedError()

    def build_ILt(self, instr): self._build_comp_op(instr)
    def build_IGt(self, instr): self._build_comp_op(instr)
    def build_ILtE(self, instr): self._build_comp_op(instr)
    def build_IGtE(self, instr): self._build_comp_op(instr)
    def build_IEq(self, instr): self._build_comp_op(instr)
    def build_INotEq(self, instr): self._build_comp_op(instr)

    def build_INeg(self, instr: INeg):
        obj = self.namespace
        name = instr.var.name

        self.constants.add(-1)
        self.emit(f"scoreboard players operation {name} {obj} *= -1 {obj}")

    def build_BMov(self, instr: BMov):
        op_type = type(instr).__name__

        if not isinstance(instr.target, BVar):
            raise NotImplementedError(f"Unsupported target operand for {op_type}")
        
        obj = self.namespace
        target = instr.target.name
        
        if isinstance(instr.source, BImm):
            self.emit(f"scoreboard players set {target} {obj} {int(instr.source.value)}")
        elif isinstance(instr.source, BVar):
            self.emit(f"scoreboard players operation {target} {obj} = {instr.source.name} {obj}")
        else:
            raise NotImplementedError

    def build_BAnd(self, instr: BAnd):
        obj = self.namespace
        target = instr.target.name

        if isinstance(instr.source, (IImm, BImm)):
            val = instr.source.value
            if val:
                pass
            else:
                self.emit(f"scoreboard players set {target} {obj} 0")
        else:
            self.emit(
                f"execute store result score {target} {obj} "
                f"unless score {target} {obj} matches 0 "
                f"unless score {instr.source.name} {obj} matches 0"
            )
    
    def build_BOr(self, instr: BOr):
        obj = self.namespace
        target = instr.target.name

        if isinstance(instr.source, (IImm, BImm)):
            val = instr.source.value
            if val:
                self.emit(f"scoreboard players set {target} {obj} 1")
            else:
                pass
        else:
            self.emit(f"execute store result score {target} {obj} unless score {target} {obj} matches 0")
            self.emit(f"execute if score {instr.source.name} {obj} matches 1 run scoreboard players set {target} {obj} 1")
    
    def build_BtoI(self, instr: BtoI):
        obj = self.namespace
        target = instr.target.name

        if isinstance(instr.source, BImm):
            self.emit(f"scoreboard players set {target} {obj} {int(instr.source.value)}")
        else:
            self.emit(f"scoreboard players operation {target} {obj} = {instr.source.name} {obj}")

    def build_ItoB(self, instr: ItoB):
        obj = self.namespace
        target = instr.target.name

        if isinstance(instr.source, IImm):
            self.emit(f"scoreboard players set {target} {obj} {int(bool(instr.source.value))}")
        else:
            self.emit(f"execute store result score {target} {obj} unless score {instr.source.name} {obj} matches 0")

    def build_BNot(self, instr: BNot):
        obj = self.namespace
        flag = instr.var.name
        self.emit(f"execute store result score {flag} {obj} if score {flag} {obj} matches 0")

    def build_Label(self, instr: Label):
        func = instr.var.name
        self.emit(f"function {self.namespace}:{func}")
        self.open_func(func)

    def build_Br(self, instr: Br):
        obj = self.namespace
        flag = instr.flag.name

        self.emit(f"execute if score {flag} {obj} matches 1 run return run function {obj}:{instr.true.name}")
        if instr.false is not None:
            self.emit(f"function {obj}:{instr.false.name}")

    def build_BrNot(self, instr: BrNot):
        obj = self.namespace
        flag = instr.flag.name

        self.emit(f"execute if score {flag} {obj} matches 0 run return run function {obj}:{instr.true.name}")
        if instr.false is not None:
            self.emit(f"function {obj}:{instr.false.name}")

    def build_Goto(self, instr: Goto):
        self.emit(f"return run function {self.namespace}:{instr.label.name}")


def validate_namespace(namespace: str):
    if namespace == "..":
        return False
    valid_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_-.")
    if len(set(namespace) | valid_chars) == len(valid_chars):
        return True
    return False


def main():
    import argparse
    import sys

    from vcs.astnodes import Module
    from vcs.irbuilder import IRBuilder
    from vcs.lexer import Lexer
    from vcs.parser import Parser
    
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
    argparser.add_argument(
        "-n",
        "--namespace",
        metavar="NAMESPACE",
        type=str,
        help="Namespace for the compiled program",
    )
    args = argparser.parse_args()

    filename: str = args.filename
    skip_comments = args.skip_comments
    namespace = args.namespace

    if namespace is None:
        namespace = filename.rpartition('.')[0]
        if not validate_namespace(namespace):
            namespace = "test"
    elif not validate_namespace(namespace):
        print(
            f"error: invalid namespace: {namespace!r}. "
            f"Minecraft namespaces can only contain lowercase letters, digits, "
            f"underscores ('_'), dashes ('-'), and dots ('.'). "
            f"Additionally, it cannot be exactly two dots ('..')."
        )
        exit(1)

    with open(filename, encoding="utf8") as f:
        source = f.read()

    errors = ErrorCollector()
    lexer = Lexer(source, filename, errors, args.tabsize)
    parser = Parser(lexer, errors, skip_comments)
    tree: Module = parser.parse()
    ir_builder = IRBuilder(errors, parser.get_error_info_on)
    ir_builder.visit(tree)
    cmd_builder = CommandBuilder(namespace, ir_builder.instructions)
    cmd_builder.build()

    if not errors.ok():
        for issue in errors.issues:
            print(dump_error(issue), file=sys.stderr)
        return

    for func, cmds in cmd_builder.mcfunctions.items():
        print(f"{func}.mcfunction:")
        if not cmds:
            print(f"  (blank)")
        for cmd in cmds:
            print(f"  {cmd}")

if __name__ == "__main__":
    main()
