import argparse
import cmd
import time
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from vcs import errors as err
from vcs import cmdgen as cgn
from vcs import irgen as irg
from vcs import lexer as lex
from vcs import parser as psr
from vcs import constfold as cf
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
            "-d",
            "--description",
            metavar="DESCRIPTION",
            help="Description for the program"
        )
        argparser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Path for the output datapack (in .zip format)"
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
            "--dump-cf",
            action="store_true",
            help="Print the generated constant-folded AST",
        )
        argparser.add_argument(
            "--dump-ir",
            action="store_true",
            help="Print the generated IR (Intermediate Representation)",
        )
        argparser.add_argument(
            "--dump-cg",
            action="store_true",
            help="Print the generated IR (Intermediate Representation)",
        )
        argparser.add_argument(
            "--dump-cmds",
            action="store_true",
            help="Print the generated datapack",
        )
        
        args = argparser.parse_args()

        filename: str | None = args.filename
        namespace: str | None = args.namespace
        description: str | None = args.description

        self.output: str = args.output

        self.skip_comments: bool = args.skip_comments
        self.tabsize: bool = args.tabsize
        self.dump_tokens: bool = args.dump_tokens
        self.lex_stats: bool = args.lex_stats
        self.dump_ast: bool = args.dump_ast
        self.dump_tree: bool = args.dump_tree
        self.parse_stats: bool = args.parse_stats
        self.dump_cf: bool = args.dump_cf
        self.dump_ir: bool = args.dump_ir
        self.dump_cg: bool = args.dump_cg
        self.dump_cmds: bool = args.dump_cmds

        stages = [
            (['dump_tokens', 'lex_stats'], self.lexer_cli, []),
            (['dump_ast', 'dump_tree', 'parse_stats'], self.parser_cli, ['vcs.lexer']),
            (['dump_cf'], self.constfold_cli, ['vcs.lexer', 'vcs.parser']),
            (['dump_ir', 'dump_cg'], self.irgen_cli, ['vcs.lexer', 'vcs.parser', 'vcs.semantic']),
            (['dump_cmds'], self.cmdgen_cli, ['vcs.lexer', 'vcs.parser', 'vcs.semantic', 'vcs.irgen']),
        ]

        funcs = []
        for flags, func, invalid_modules in stages:
            if any(getattr(self, flag) for flag in flags):
                if modname in invalid_modules:
                    flags_str = ','.join(f'--{flag}' for flag in flags if getattr(self, flag))
                    utils.print_error(f"{flags_str} cannot be used in {modname!r}")
                    exit(1)
                funcs.append(func)

        func: Callable[[str, str], None] | None = {
            "vcs.lexer": self.lexer_cli,
            "vcs.parser": self.parser_cli,
            "vcs.constfold": self.constfold_cli,
            "vcs.semantic": self.semantic_cli,
            "vcs.irgen": self.irgen_cli,
            "vcs.cmdgen": self.cmdgen_cli,
            "vcs.cli": self.main_cli,
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
        self.description: str = description or self.namespace

        with open(path, encoding="utf8") as f:
            source = f.read()
        
        cli(filename, source)

    def build_pipeline(self, filename: str, source: str):
        self.errors = err.ErrorCollector()
        lexer = lex.Lexer(source, filename, self.errors, self.tabsize)
        parser = psr.Parser(lexer, self.errors, self.skip_comments, self.dump_tree)
        constfolder = cf.ConstantFolder(parser)
        typechecker = sem.SemanticAnalyzer(constfolder, self.errors)
        irgen = irg.IRGenerator(self.namespace, typechecker)
        cmdgen = cgn.CommandGenerator(irgen)
        return (lexer, parser, constfolder, typechecker, irgen, cmdgen)

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
                f"Lexer Duration: {dt:.3f} sec ({nlines} lines{line_psec if dt else ''}); Tokens: {len(tokens)}"
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
                f"Parser Duration: {dt:.3f} sec ({nlines} lines{line_psec if dt else ''}); Cache size: {len(parser._cache)}"
            )
        
        self.print_errors_if_any()

    def constfold_cli(self, filename: str, source: str):
        _, _, constfolder, *_ = self.build_pipeline(filename, source)
        folded = constfolder.fold()
        if self.dump_cf:
            utils.print_info(str(folded))
        self.print_errors_if_any()

    def semantic_cli(self, filename: str, source: str):
        _, _, _, typechecker, *_ = self.build_pipeline(filename, source)
        typechecker.analyze()
        self.print_errors_if_any()
    
    def irgen_cli(self, filename: str, source: str):
        _, _, _, _, irgen, *_ = self.build_pipeline(filename, source)
        mod = irgen.generate()
        if mod is not None and (self.dump_ir or self.dump_cg):
            if self.dump_ir:
                utils.print_info(str(mod))
                
            if self.dump_cg:
                call_graph = mod.build_call_graph()
                for src, dsts in call_graph.func_calls.items():
                    utils.print_info(f"{src.name} -> {', '.join(dst.name for dst in dsts)}")

        self.print_errors_if_any()
    
    def cmdgen_cli(self, filename: str, source: str):
        _, _, _, _, _, cmdgen, *_ = self.build_pipeline(filename, source)
        datapack = cmdgen.generate()
        if datapack is not None and self.dump_cmds:
            utils.print_info(str(datapack))
        self.print_errors_if_any()
    
    def main_cli(self, filename: str, source: str):
        _, _, _, _, _, cmdgen, *_ = self.build_pipeline(filename, source)
        datapack = cmdgen.generate()
        if datapack is not None:
            datapack.write_zip(self.output, self.description)

        self.print_errors_if_any()


def cli():
    frame = sys._getframe(1)
    module = frame.f_globals.get("__name__")
    assert module is not None
    CommandLineInterface(sys.modules[module])


if __name__ == "__main__":
    cli()