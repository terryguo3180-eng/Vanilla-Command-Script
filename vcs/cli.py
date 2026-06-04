import argparse
import cmd
import time
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from vcs import errors as err
from vcs import irgen as irg
from vcs import lexer as lex
from vcs import parser as psr
from vcs import semantic as sem
from vcs import utils


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
        argparser.add_argument(dest="filename", nargs="?", metavar="filename.vcs")
        argparser.add_argument(
            "-n",
            "--namespace",
            metavar="NAMESPACE",
            help="Namespace for the program"
        )
        argparser.add_argument(
            "--skip-comments",
            action="store_true",
            help="Skip all the comment tokens",
        )
        argparser.add_argument(
            "--tabsize",
            metavar="TABSIZE",
            default=4,
            help="How many spaces per tab character",
        )
        argparser.add_argument(
            "--dump-tokens",
            action="store_true",
            help="Print the generated tokens"
        )
        argparser.add_argument(
            "--lex-stats",
            action="store_true",
            help="Print stats of the lexer",
        )
        argparser.add_argument(
            "--dump-tree",
            action="store_true",
            help="Print the complete parse tree of the parser",
        )
        argparser.add_argument(
            "--dump-ast",
            action="store_true",
            help="Print the generated AST (Abstract Syntax Tree)",
        )
        argparser.add_argument(
            "--parse-stats",
            action="store_true",
            help="Print stats of the parser",
        )
        argparser.add_argument(
            "--dump-ir",
            action="store_true",
            help="Print the generated IR (Intermediate Representation)",
        )
        
        args = argparser.parse_args()

        filename: str | None = args.filename
        namespace: str | None = args.namespace

        self.skip_comments: bool = args.skip_comments
        self.tabsize: bool = args.tabsize
        self.dump_tokens: bool = args.dump_tokens
        self.lex_stats: bool = args.lex_stats
        self.dump_ast: bool = args.dump_ast
        self.dump_tree: bool = args.dump_tree
        self.parse_stats: bool = args.parse_stats
        self.dump_ir: bool = args.dump_ir

        funcs = []
        if self.dump_tokens or self.lex_stats:
            funcs.append(self.lexer_cli)
        if self.dump_ast or self.dump_tree or self.parse_stats:
            if modname == "vcs.lexer":
                utils.print_error(f"Parser arguments cannot be used in {modname!r}")
                exit(1)
            funcs.append(self.parser_cli)
        if self.dump_ir:
            if modname in ["vcs.lexer", "vcs.parser", "vcs.semantic"]:
                utils.print_error(f"IR generator arguments cannot be used in {modname!r}")
                exit(1)
            funcs.append(self.irgen_cli)

        func: Callable[[str, str], None] | None = {
            "vcs.lexer": self.lexer_cli,
            "vcs.parser": self.parser_cli,
            "vcs.semantic": self.semantic_cli,
            "vcs.irgen": self.irgen_cli,
            "vcs.cli": self.irgen_cli,
        }.get(modname)

        if func is None:
            raise RuntimeError(f"Unknown module: {modname!r}")

        if func not in funcs:
            funcs.append(func)
        
        def cli(filename: str, source: str):
            for func in funcs:
                func(filename, source)

        if filename is None:
            self.namespace = namespace or "repl"
            intro = utils.color_text(
                f"{modname} REPL (Read-Eval-Print Loop) module, type Ctrl+C to exit the program",
                utils.AnsiCode.MAGENTA, bold=True
            )
            REPL(cli).cmdloop(intro)
            exit(0)

        path = Path(filename)
        if not path.is_file():
            utils.print_error(f"Source file {filename!r} does not exist")
            exit(1)

        self.filename: str = filename
        self.namespace: str = namespace or path.stem

        with open(path, encoding="utf8") as f:
            source = f.read()
        
        cli(filename, source)

    def build_pipeline(self, filename: str, source: str):
        self.errors = err.ErrorCollector()
        lexer = lex.Lexer(source, filename, self.errors, self.tabsize)
        parser = psr.Parser(lexer, self.errors, self.skip_comments, self.dump_tree)
        typechecker = sem.SemanticAnalyzer(parser, self.errors)
        irgen = irg.IRGenerator(self.namespace, typechecker)
        return (lexer, parser, typechecker, irgen)

    def print_errors_if_any(self):
        if not self.errors.ok():
            self.errors.sort()
            for issue in self.errors.issues:
                utils.print_compiler_error(issue)

    def lexer_cli(self, filename: str, source: str):
        t0 = time.time()
        lexer, *_ = self.build_pipeline(filename, source)
        tokens = list(lexer)
        t1 = time.time()

        if self.dump_tokens:
            for token in tokens:
                utils.print_info(str(token))

        if self.lex_stats:
            dt = t1 - t0
            nlines = tokens[-1].end_lineno
            line_psec = f"; {nlines / dt:.0f} lines/sec"
            utils.print_info(
                f"Lexer Duration: {dt:.3f} sec ({nlines} lines{line_psec if dt else ''}); "
                f"Tokens: {len(tokens)}"
            )

        self.print_errors_if_any()

    def parser_cli(self, filename: str, source: str):
        t0 = time.time()
        _, parser, *_ = self.build_pipeline(filename, source)
        tree = parser.parse()
        t1 = time.time()

        if self.dump_ast:
            utils.print_info(str(tree))
        
        if self.parse_stats:
            dt = t1 - t0
            nlines = parser.diagnose().end_lineno
            line_psec = f", {nlines / dt:.0f} lines/sec"
            utils.print_info(
                f"Parser Duration: {dt:.3f} sec ({nlines} lines{line_psec if dt else ''}); "
                f"Tokens: {len(parser._tokenstream._tokens)}; Cache size: {len(parser._cache)}"
            )
        
        self.print_errors_if_any()

    def semantic_cli(self, filename: str, source: str):
        _, _, typechecker, *_ = self.build_pipeline(filename, source)
        typechecker.analyze()
        self.print_errors_if_any()
    
    def irgen_cli(self, filename: str, source: str):
        _, _, _, irgen, *_ = self.build_pipeline(filename, source)
        module = irgen.generate()
        if module is not None and self.dump_ir:
            utils.print_info(str(module))
        self.print_errors_if_any()


def cli():
    frame = sys._getframe(1)
    module = frame.f_globals.get("__name__")
    assert module is not None
    CommandLineInterface(sys.modules[module])
