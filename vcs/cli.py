import argparse
import cmd
import sys

from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from vcs import errors as err
from vcs import cmdgen as cgn
from vcs import irgen as irg
from vcs import lexer as lex
from vcs import parser as psr
from vcs import constfold as cf
from vcs import semantic as sem
from vcs import utils


ARG_SPECS = [
    {
        "dest": "filename",
        "nargs": "?",
        "metavar": "filename.vcs",
        "condition": lambda m: True
    }, {
        "dest": "namespace",
        "args": ["-n", "--namespace"],
        "metavar": "NAMESPACE",
        "help": "Namespace for the program",
        "condition": lambda m: True
    }, {
        "dest": "skip_comments",
        "args": ["-k", "--skip-comments"],
        "action": "store_true",
        "help": "Ignore all the comments",
        "condition": lambda m: True
    }, {
        "dest": "tabsize",
        "args": ["-t", "--tabsize"],
        "metavar": "TABSIZE",
        "default": 4,
        "help": "How many spaces per tab character",
        "condition": lambda m: True
    }, {
        "dest": "lineno_digits",
        "args": ["-l", "--lineno-digits"],
        "metavar": "LINENO_DIGITS",
        "help": "Max digits for line numbers in mcfunction names",
        "condition": lambda m: True
    }, {
        "dest": "dump_tokens",
        "args": ["--dump-tokens"],
        "action": "store_true",
        "help": "Print the generated tokens",
        "condition": lambda m: True
    }, {
        "dest": "description",
        "args": ["-d", "--description"],
        "metavar": "DESCRIPTION",
        "help": "Description for the program",
        "condition": lambda m: m == "vcs.cli"
    },
    {
        "dest": "output",
        "args": ["-o", "--output"],
        "required": True,
        "help": "Path for the output datapack (in .zip format)",
        "condition": lambda m: m == "vcs.cli"
    }, {
        "dest": "fixed_precision",
        "args": ["-x", "--fixed-precision"],
        "metavar": "PRECISION",
        "type": int,
        "default": 1000,
        "help": "Precision for fixed-point numbers (default: 1000)",
        "condition": lambda m: m not in {
            "vcs.lexer", "vcs.parser", "vcs.constfold", "vcs.semantic", "vcs.irgen"
        }
    }, {
        "dest": "dump_tree",
        "args": ["--dump-tree"],
        "action": "store_true",
        "help": "Print the complete parse tree of the parser",
        "condition": lambda m: m != "vcs.lexer"
    }, {
        "dest": "dump_ast",
        "args": ["--dump-ast"],
        "action": "store_true",
        "help": "Print the generated AST (Abstract Syntax Tree)",
        "condition": lambda m: m != "vcs.lexer"
    }, {
        "dest": "parse_stats",
        "args": ["--parse-stats"],
        "action": "store_true",
        "help": "Print stats of the parser",
        "condition": lambda m: m != "vcs.lexer"
    }, {
        "dest": "dump_cf",
        "args": ["--dump-cf"],
        "action": "store_true",
        "help": "Print the generated constant-folded AST",
        "condition": lambda m: m not in {"vcs.lexer", "vcs.parser"}
    }, {
        "dest": "dump_ir",
        "args": ["--dump-ir"],
        "action": "store_true",
        "help": "Print the generated IR (Intermediate Representation)",
        "condition": lambda m: m not in {
            "vcs.lexer", "vcs.parser", "vcs.constfold", "vcs.semantic"
        }
    }, {
        "dest": "dump_cg",
        "args": ["--dump-cg"],
        "action": "store_true",
        "help": "Print the generated call graph",
        "condition": lambda m: m not in {
            "vcs.lexer", "vcs.parser", "vcs.constfold", "vcs.semantic"
        }
    }, {
        "dest": "dump_cmds",
        "args": ["--dump-cmds"],
        "action": "store_true",
        "help": "Print the generated datapack",
        "condition": lambda m: m not in {
            "vcs.lexer", "vcs.parser", "vcs.constfold", "vcs.semantic", "vcs.irgen"
        }
    },
]


main_module = sys.modules[__name__]

class REPL(cmd.Cmd):
    def __init__(self, cli: Callable[[str, str], None], linecont=True):
        super().__init__()
        self.cli = cli
        self.linecont = linecont
        self.prompt = "[In 0] "
        self.input_n = 0
        self.multiline = False
        self.lastline = ""

    def emptyline(self):
        return False

    def precmd(self, line: str) -> str:
        self.lastline = line
        return line

    def do_help(self, arg: str) -> None:
        return self.default(self.lastline)

    def default(self, line: str) -> None:
        if not line:
            return
        if self.linecont and line.endswith('\\'):
            line = line[:-1]
            # Line continuation
            while True:
                new = input("." * (len(self.prompt) - 1) + " ")
                line += "\n"
                if not new:
                    break
                line += new

        self.cli(self.prompt[:-1], line)
        
        self.input_n += 1
        self.prompt = f"[In {self.input_n}] "

    def quit(self) -> bool:
        return True

    def cmdloop(self, intro=None) -> None:
        try:
            super().cmdloop(intro)
        except KeyboardInterrupt:
            pass


class CommandLineInterface:
    def __init__(self, module: ModuleType = main_module):
        self.module = module
        if module.__name__ == "__main__":
            assert module.__spec__ is not None
            modname = module.__spec__.name
        else:
            modname = module.__name__

        argparser = argparse.ArgumentParser(prog=f"python -m {modname}")
        for spec in ARG_SPECS:
            if spec["condition"](modname):
                args_list = spec.get("args")
                kwargs = {k: v for k, v in spec.items() if k not in ("dest", "args", "condition")}
                if args_list is None:
                    argparser.add_argument(spec["dest"], **kwargs)
                else:
                    argparser.add_argument(*args_list, **kwargs)

        args = argparser.parse_args()

        for spec in ARG_SPECS:
            if spec["condition"](modname):
                setattr(self, spec["dest"], getattr(args, spec["dest"]))

        filename: str | None = args.filename
        namespace: str | None = args.namespace
        lineno_digits: int | None = args.lineno_digits

        func: Callable[[str, str], None] | None = getattr(self, modname.removeprefix("vcs.") + "_cli")
        if func is None:
            raise RuntimeError(f"Unknown module: {modname!r}")

        if filename is None:
            self.namespace = namespace or "repl"
            intro = utils.color_text(
                f"{modname} REPL (Read-Eval-Print Loop) module, type Ctrl+C to exit the program",
                utils.AnsiCode.MAGENTA, bold=True
            )
            REPL(func).cmdloop(intro)
            exit(0)

        path = Path(filename)
        if not path.is_file():
            utils.print_error(f"Source file {filename!r} does not exist")
            exit(1)

        self.filename = filename
        self.namespace = namespace or path.stem

        if modname == "vcs.cli":
            self.description = self.description or self.namespace

        with open(path, encoding="utf8") as f:
            source = f.read()

        self.lineno_digits = lineno_digits or len(str(source.count("\n") + 1))

        func(filename, source)

    # Implemented __getattr__ just to make the type checker happy
    def __getattr__(self, name: str) -> Any:
        value = self.__dict__.get(name, ...)
        if value is ...:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return value

    def build_pipeline(self, filename: str, source: str, name: str) -> Any:
        self.errors = err.ErrorCollector()
        lexer = lex.Lexer(source, filename, self.errors, self.skip_comments, self.tabsize, self.dump_tokens)
        if name == "lexer":
            return lexer

        parser = psr.Parser(lexer, self.errors, self.dump_tree, self.dump_ast, self.parse_stats)
        if name == "parser":
            return parser
        
        constfolder = cf.ConstantFolder(parser, self.dump_cf)
        if name == "constfold":
            return constfolder

        typechecker = sem.SemanticAnalyzer(constfolder, self.errors)
        if name == "semantic":
            return typechecker

        irgen = irg.IRGenerator(self.namespace, typechecker, self.lineno_digits, self.dump_ir, self.dump_cg)
        if name == "irgen":
            return irgen

        cmdgen = cgn.CommandGenerator(irgen, self.fixed_precision, self.dump_cmds)
        if name == "cmdgen":
            return cmdgen

    def print_errors_if_any(self):
        if not self.errors.ok():
            self.errors.sort()
            for issue in self.errors.issues:
                utils.print_compiler_error(issue)

    def lexer_cli(self, filename: str, source: str):
        lexer = self.build_pipeline(filename, source, "lexer")
        list(lexer)
        self.print_errors_if_any()

    def parser_cli(self, filename: str, source: str):
        parser = self.build_pipeline(filename, source, "parser")
        parser.parse()
        self.print_errors_if_any()

    def constfold_cli(self, filename: str, source: str):
        constfolder = self.build_pipeline(filename, source, "constfold")
        constfolder.fold()
        self.print_errors_if_any()

    def semantic_cli(self, filename: str, source: str):
        typechecker = self.build_pipeline(filename, source, "semantic")
        typechecker.analyze()
        self.print_errors_if_any()
    
    def irgen_cli(self, filename: str, source: str):
        irgen= self.build_pipeline(filename, source, "irgen")
        irgen.generate()
        self.print_errors_if_any()
    
    def cmdgen_cli(self, filename: str, source: str):
        cmdgen = self.build_pipeline(filename, source, "cmdgen")
        cmdgen.generate()
        self.print_errors_if_any()
    
    def main_cli(self, filename: str, source: str):
        cmdgen = self.build_pipeline(filename, source, "cmdgen")
        datapack = cmdgen.generate()
        if datapack is not None:
            datapack.write_zip(self.output, self.description)


def cli():
    frame = sys._getframe(1)
    module = frame.f_globals.get("__name__")
    assert module is not None
    CommandLineInterface(sys.modules[module])


if __name__ == "__main__":
    cli()
