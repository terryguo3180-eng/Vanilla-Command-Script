import cmd
from typing import Callable


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
