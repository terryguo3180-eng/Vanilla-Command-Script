import argparse
import json
import sys
import zipfile

from vcs.astnodes import Module
from vcs.errors import *
from vcs.lexer import Lexer
from vcs.parser import Parser
from vcs.ir import *
from vcs.irbuilder import IRBuilder
from vcs.cmdbuilder import CommandBuilder, validate_namespace


argparser = argparse.ArgumentParser()
argparser.add_argument(dest="filename", metavar="filename.vcs")
argparser.add_argument(
    "-o",
    "--output",
    metavar="OUTPUT",
    required=True,
    type=str,
    help="Output datapack file (.zip)",
)
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
argparser.add_argument(
    "-d",
    "--description",
    metavar="DESCRIPTION",
    type=str,
    default="",
    help="Description of the datapack"
)
args = argparser.parse_args()

filename: str = args.filename
output: str = args.output
tabsize: int = args.tabsize
skip_comments: bool = args.skip_comments
namespace: str = args.namespace
description: str = args.description

if not output.endswith('.zip'):
    print(f"error: output datapack must be in .zip format")
    exit(1)

if namespace is None:
    namespace = filename.rpartition('.')[0]
    if not validate_namespace(namespace):
        namespace = "test"
elif not validate_namespace(namespace):
    print(
        f"error: invalid namespace: {namespace!r}. "
        f"Minecraft namespaces can only contain lowercase letters, digits, "
        f"underscores ('_'), dashes ('-'), and dots ('.'). "
        f"Additionally, it cannot be exactly '..'."
    )
    exit(1)

with open(filename, encoding="utf8") as f:
    source = f.read()

errors = ErrorCollector()
lexer = Lexer(source, filename, errors, tabsize)
parser = Parser(lexer, errors, skip_comments)
tree: Module = parser.parse()
ir_builder = IRBuilder(errors, parser.get_error_info_on)
ir_builder.visit(tree)
cmd_builder = CommandBuilder(namespace, ir_builder.instructions)
cmd_builder.build()

if not errors.ok():
    for issue in errors.issues:
        print(dump_error(issue), file=sys.stderr)
    exit(1)

with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as f:
    pack_mcmeta = {
        "pack": {
            "description": description,
            "min_format": 94,
            "max_format": 94,
        }
    }
    f.writestr("pack.mcmeta", json.dumps(pack_mcmeta, indent=2))
    for func, cmds in cmd_builder.mcfunctions.items():
        f.writestr(f"data/{namespace}/function/{func}.mcfunction", "\n".join(cmds))
