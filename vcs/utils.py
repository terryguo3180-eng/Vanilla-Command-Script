import os
import re
import sys

from vcs import errors as err


class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AnsiCode:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    PURPLE = "\033[95m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BGRED = "\033[41m"
    CODE_BG = "\033[48;2;35;35;35m"
    CURLY_U = "\033[4:3m"


def _get_terminal_size() -> os.terminal_size | None:
    try:
        return os.get_terminal_size()
    except OSError:
        return None

def color_text(text: str, color: str = AnsiCode.RESET, bold: bool = False) -> str:
    return (AnsiCode.BOLD if bold else "") + color + text + AnsiCode.RESET

def print_error(text: str) -> None:
    if _get_terminal_size() is None:
        print(f"[ERROR] {text}", file=sys.stderr)
        return
    print(color_text(text, AnsiCode.RED, bold=True), file=sys.stderr)

def print_success(text: str) -> None:
    if _get_terminal_size() is None:
        print(f"[INFO] {text}")
        return
    print(color_text(text, AnsiCode.GREEN))

def print_info(text: str) -> None:
    if _get_terminal_size() is None:
        print(f"[INFO] {text}")
        return
    print(text)

def print_warning(text: str) -> None:
    if _get_terminal_size() is None:
        print(f"[WARNING] {text}", file=sys.stderr)
        return
    print(color_text(text, AnsiCode.YELLOW), file=sys.stderr)


def print_compiler_error(e: err.CompilerError):
    info = e.info
    fname = info.filename
    l1 = info.lineno
    c1 = info.column
    l2 = info.end_lineno
    c2 = info.end_column
    
    title = "warning" if e.warning else "error"
    code = e.code
    msg = e.message
    cnt_val = str(err.counter() - 1)
    cnt_digits = len(cnt_val)
    fmt_code = "CE" + str(code).zfill(cnt_digits)

    tsize = _get_terminal_size()
    if tsize is None:
        plain = ''.join((
            "[", title, "]: ", fname, ":",
            str(l1), ",", str(c1), "-", str(l2), ",", str(c2),
            ": ", msg, " (", fmt_code, ")"
        ))
        print(plain, file=sys.stderr)
        return
    
    HIGHLIGHT = AnsiCode.CURLY_U + (AnsiCode.YELLOW if e.warning else AnsiCode.RED)
    DESCR_COLOR = AnsiCode.GREEN if e.warning else AnsiCode.PURPLE

    lines = info.source.split("\n")
    total = len(lines)
    w = len(str(total))
    affected = lines[l1 - 1:l2]
    first = affected[0]
    last = affected[-1]
    width = tsize.columns
    
    # Highlight error info in source code
    if len(affected) == 1:
        before = first[:c1]
        error_sec = first[c1:c2]
        after = first[c2:]
        affected[0] = ''.join((
            AnsiCode.RESET, AnsiCode.CODE_BG, before, HIGHLIGHT,
            error_sec, AnsiCode.RESET, AnsiCode.CODE_BG, after,
        ))
    else:
        # Multi-line error: highlight first line
        before = first[:c1]
        from_err_to_end = first[c1:] or " "
        affected[0] = ''.join((
            AnsiCode.RESET, AnsiCode.CODE_BG, before, HIGHLIGHT,
            from_err_to_end, AnsiCode.RESET, AnsiCode.CODE_BG,
        ))

        # Newline / endmarker
        if c2 == 0 and (
            len(affected) == 2 or all(line == "" for line in affected[1:-1])
        ):
            affected = [affected[0]]
        else:
            # Highlight last line
            up_to_err = last[:c2]
            after = last[c2:]
            affected[-1] = ''.join((
                HIGHLIGHT, up_to_err, AnsiCode.RESET, AnsiCode.CODE_BG, after,
            ))

        # Highlight middle lines entirely
        for i, line in enumerate(affected[1:-1], start=1):
            affected[i] = AnsiCode.CURLY_U + AnsiCode.RED + line + AnsiCode.RESET

    # Format each error line with line numbers
    for i, line in enumerate(affected):
        cur = l1 + i
        prefix_num = " " + str(cur).rjust(w) + " │ "
        prefix = AnsiCode.CODE_BG + AnsiCode.GRAY + prefix_num + AnsiCode.WHITE
        
        # Calculate padding to fill console width
        ansi_pat = re.compile(r'\x1b\[[0-9;]*m')
        real_len = len(ansi_pat.sub("", line))
        pad = max(0, width - 4 - w - real_len)
        postfix = AnsiCode.CODE_BG + " " * pad
        affected[i] = prefix + line + postfix + AnsiCode.RESET

    # Build location string
    loc = ''.join((
        AnsiCode.GRAY, fname, AnsiCode.RESET, ":",
        AnsiCode.BLUE, str(l1), AnsiCode.RESET, ":",
        AnsiCode.BLUE, str(c1), AnsiCode.RESET, "-",
        AnsiCode.BLUE, str(l2), AnsiCode.RESET, ":",
        AnsiCode.BLUE, str(c2), AnsiCode.WHITE, ": ",
    ))
    
    title_and_msg = ''.join((
        DESCR_COLOR, AnsiCode.BOLD, title, AnsiCode.RESET, ": ",
        DESCR_COLOR, msg, " ",
        AnsiCode.YELLOW, "(", fmt_code, ")", AnsiCode.RESET
    ))
    
    out = loc + title_and_msg + "\n"
    out += "\n".join(affected)

    print(out, file=sys.stderr)


class CompareOp(metaclass=SingletonMeta): ...


class EqOp(CompareOp):
    def __str__(self): return "=="

class NotEqOp(CompareOp):
    def __str__(self): return "!="

class GtOp(CompareOp):
    def __str__(self): return ">"

class LtOp(CompareOp):
    def __str__(self): return "<"

class GtEOp(CompareOp):
    def __str__(self): return ">="

class LtEOp(CompareOp):
    def __str__(self): return "<="
