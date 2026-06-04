from __future__ import annotations

import re
import os
from dataclasses import dataclass
from itertools import count

from vcs.ansi import AnsiCode


@dataclass(frozen=True)
class ErrorInfo:
    filename: str
    lineno: int
    column: int
    source: str
    end_lineno: int
    end_column: int


@dataclass
class ErrorCode:
    code: int
    message: str
    
    def format(self, *args, **kwargs) -> ErrorCode:
        return ErrorCode(self.code, self.message.format(*args, **kwargs))


class ErrorCollector:
    def __init__(self):
        self.issues: list[CompilerError] = []
    
    def add(self, issue: CompilerError):
        self.issues.append(issue)
    
    def ok(self):
        return not any(not issue.warning for issue in self.issues)
    
    def sort(self, reverse=False):
        self.issues.sort(key=lambda e: (e.info.lineno, e.info.column), reverse=reverse)


counter = count().__next__


class CompilerError(Exception):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        self.info = info
        self.warning = warning
        self.message = "Unknown"


class LexError(CompilerError):
    code = counter()
    
    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Invalid syntax"


class CharAfterLineContinuation(LexError):
    code = counter()
    
    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Unexpected character after line continuation character"


class UnexpectedEOF(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Unexpected EOF after line continuation character"


class InvalidHexEscape(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Invalid hex escape"


class InvalidEscapeSequence(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, char: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Invalid escape sequence: '\\{char}'"


class UnterminatedString(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Unterminated string literal"


class UnexpectedIndent(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Unexpected indent"


class MismatchedUnindent(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Unindent does not match any outer indentation level"


class MismatchedBracket(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, bracket: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Unmatched '{bracket}'"


class UnclosedBracket(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, bracket: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Unclosed '{bracket}'"


class InvalidCharacter(LexError):
    code = counter()

    def __init__(self, info: ErrorInfo, char: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Invalid character '{char}'"


class ParseError(CompilerError):
    code = counter()
    
    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Invalid syntax"


class BreakOutsideLoop(ParseError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Break statement outside loop"


class ContinueOutsideLoop(ParseError):
    code = counter()

    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Continue statement outside loop"


class ExpectationFailed(ParseError):
    code = counter()

    def __init__(self, info: ErrorInfo, expected: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Expected {expected}"


class SemanticError(CompilerError):
    code = counter()
    
    def __init__(self, info: ErrorInfo, warning=False):
        super().__init__(info, warning)
        self.message = "Invalid semantic"


class FunctionDeclared(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, name: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Function '{name}' has already been declared"


class FunctionNotDeclared(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, name: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Undefined function '{name}'"


class VariableDeclared(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, name: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Variable '{name}' has already been declared"


class VariableNotDeclared(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, name: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Undefined identifier '{name}'"


class UnassignableType(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, actual: str, expected: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Type '{actual}' is not assignable to declared type '{expected}'"


class InvalidAugmentedOpTypes(SemanticError):
    code = counter()

    def __init__(
        self, info: ErrorInfo, left_type: str, right_type: str, op: str, warning=False
    ):
        super().__init__(info, warning)
        self.message = f"Invalid operand types for augmented '{op}': '{left_type}' and '{right_type}'"


class InvalidBinaryOpTypes(SemanticError):
    code = counter()

    def __init__(
        self, info: ErrorInfo, left_type: str, right_type: str, op: str, warning=False
    ):
        super().__init__(info, warning)
        self.message = f"Invalid operand types for binary '{op}': '{left_type}' and '{right_type}'"


class InvalidUnaryOpType(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, op_type: str, op: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Invalid operand type for unary '{op}': '{op_type}'"


class InvalidAssignment(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, target: str, warning=False):
        super().__init__(info, warning)
        self.message = f"{target} can't be used for assignment"


class InvalidType(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, source_type: str, target_type: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Type '{source_type}' is not assignable to type '{target_type}'"


class InvalidUnaryType(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, operator: str, actual: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Invalid type for unary '{operator}': '{actual}'"


class ExcessPosArgs(SemanticError):
    code = counter()

    def __init__(
        self, info: ErrorInfo, func_name: str, expected: int, given: int, warning=False
    ):
        super().__init__(info, warning)
        plural = "s" if expected != 1 else ""
        self.message = f"'{func_name}' takes {expected} positional argument{plural} but {given} {'were' if given != 1 else 'was'} given"


class ExcessPosArgsDefault(SemanticError):
    code = counter()

    def __init__(
        self,
        info: ErrorInfo,
        func_name: str,
        min_expected: int,
        max_expected: int,
        given: int,
        warning=False
    ):
        super().__init__(info, warning)
        plural = "s" if max_expected != 1 else ""
        self.message = f"'{func_name}' takes from {min_expected} to {max_expected} positional argument{plural} but {given} {'were' if given != 1 else 'was'} given"


class MissingParams(SemanticError):
    code = counter()

    def __init__(
        self, info: ErrorInfo, func_name: str, missing_count: int, missing_params: list[str], warning=False
    ):
        super().__init__(info, warning)
        plural = "s" if missing_count != 1 else ""
        self.message = f"'{func_name}' missing {missing_count} required positional argument{plural}: {", ".join(missing_params)}"


class DuplicateArgument(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, param_name: str, warning=False):
        super().__init__(info, warning)
        self.message = f"Duplicate argument: {param_name}"


class UnexpectedKwarg(SemanticError):
    code = counter()

    def __init__(self, info: ErrorInfo, func_name: str, kwargs: list[str], warning=False):
        super().__init__(info, warning)
        if len(kwargs) == 1:
            self.message = f"'{func_name}' got an unexpected keyword argument: '{kwargs[0]}'"
            return
        self.message = f"'{func_name}' got unexpected keyword arguments: '{", ".join(kwargs)}'"


def dump_error(error: CompilerError, fancy=True):
    try:
        console_width = os.get_terminal_size().columns
    except OSError:
        console_width = 0  # Only for type checker
        fancy = False

    info = error.info
    filename, lineno, column, end_lineno, end_column = (
        info.filename, info.lineno, info.column, info.end_lineno, info.end_column
    )
    title = "warning" if error.warning else "error"
    errcode, message = error.code, error.message
    errcode_digits = len(str(counter() - 1))
    codestr = f"CE{errcode:0>{errcode_digits}}"

    if not fancy:
        return f"[{title}]: {filename}:{lineno},{column}-{end_lineno},{end_column}: {message} ({codestr})"

    highlight_theme =  AnsiCode.CURLY_U + (AnsiCode.YELLOW if error.warning else AnsiCode.RED)
    descr_color = AnsiCode.GREEN if error.warning else AnsiCode.PURPLE

    lines = info.source.split("\n")
    total_lines = len(lines)
    lineno_digits = len(str(total_lines))
    error_lines = lines[lineno - 1:end_lineno]
    first_line = error_lines[0]
    last_line = error_lines[-1]

    # Highlight error info in source code
    if len(error_lines) == 1:
        error_lines[0] = (
            f"{AnsiCode.RESET}{AnsiCode.CODE_BACKGROUND}{first_line[:column]}"
            f"{highlight_theme}{first_line[column:end_column]}"
            f"{AnsiCode.RESET}{AnsiCode.CODE_BACKGROUND}{first_line[end_column:]}"
        )
    else:
        # Multi-line error: highlight first line
        error_lines[0] = (
            f"{AnsiCode.RESET}{AnsiCode.CODE_BACKGROUND}{first_line[:column]}"
            f"{highlight_theme}{first_line[column:] or " "}"
            f"{AnsiCode.RESET}{AnsiCode.CODE_BACKGROUND}"
        )

        # Newline / endmarker
        if end_column == 0 and (
            len(error_lines) == 2 or all(line == "" for line in error_lines[1:-1])
        ):
            error_lines = [error_lines[0]]
        else:
            # Highlight last line
            error_lines[-1] = (
                f"{highlight_theme}{last_line[:end_column]}"
                f"{AnsiCode.RESET}{AnsiCode.CODE_BACKGROUND}{last_line[end_column:]}"
            )

        # Highlight middle lines entirely
        for i, line in enumerate(error_lines[1:-1], start=1):
            error_lines[i] = AnsiCode.CURLY_U + AnsiCode.RED + line + AnsiCode.RESET

    # Format each error line with line numbers
    for i, line in enumerate(error_lines):
        prefix = (
            f"{AnsiCode.CODE_BACKGROUND}{AnsiCode.GRAY} {lineno+i: >{lineno_digits}}"
            f" │ {AnsiCode.WHITE}"
        )
        # Calculate padding to fill console width
        real_len = len(re.compile(r'\x1b\[[0-9;]*m').sub("", line))  # Length without ansi escapes
        padding = max(0, console_width - 4 - lineno_digits - real_len)
        postfix = AnsiCode.CODE_BACKGROUND + " " * padding
        error_lines[i] = f"{prefix}{line}{postfix}{AnsiCode.RESET}"

    # Print error message with info and formatted source
    output = (
        f"{AnsiCode.GRAY}{filename}{AnsiCode.RESET}:{AnsiCode.BLUE}{lineno}"
        f"{AnsiCode.RESET}:{AnsiCode.BLUE}{column}{AnsiCode.RESET}-{AnsiCode.BLUE}"
        f"{end_lineno}{AnsiCode.RESET}:{AnsiCode.BLUE}{end_column}{AnsiCode.WHITE}: "
        f"{descr_color}{AnsiCode.BOLD}{title}{AnsiCode.RESET}: {descr_color}{message} "
        f"{AnsiCode.YELLOW}({codestr}){AnsiCode.RESET}\n"
    )
    output += "\n".join(error_lines)
    return output
