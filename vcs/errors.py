from __future__ import annotations

from dataclasses import dataclass
from itertools import count


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
        verb = 'were' if given != 1 else 'was'
        self.message = f"'{func_name}' takes {expected} positional argument{plural} but {given} {verb} given"


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
        verb = 'were' if given != 1 else 'was'
        self.message = f"'{func_name}' takes from {min_expected} to {max_expected} positional argument{plural} but {given} {verb} given"


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

