from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator

from vcs import errors as err
from vcs import utils


# Small helpers to build regex components
def _group(*choices):
    return "(" + "|".join(choices) + ")"

def _maybe(*choices):
    return _group(*choices) + "?"


class LexerConfig:
    """
    Configuration class that contains various mappings and regexes used in the lexer.
    """

    brackets = {"(": ")", "[": "]", "{": "}"}
    operators = "( ) [ ] { } : , ; + - * / < > = . % @ := != <= >= == -> += -= *= /= %=".split()
    simple_escapes = {"\\": "\\", "n": "\n", "t": "\t"}
    hex_escapes = {"x": 2, "u": 4, "U": 8}

    # Regexes adapted from the `tokenize` standard library with slight modifications
    re_Whitespace = r" +"
    re_Comment = r"#[^\r\n]*"
    re_Name = r"\w+"
    re_Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
    re_Binnumber = r"0[bB](?:_?[01])+"
    re_Octnumber = r"0[oO](?:_?[0-7])+"
    re_Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
    re_Intnumber = _group(re_Hexnumber, re_Binnumber, re_Octnumber, re_Decnumber)
    re_Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
    re_Pointfloat = _group(r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*") + _maybe(re_Exponent)
    re_Expfloat = r"[0-9](?:_?[0-9])*" + re_Exponent
    re_Floatnumber = _group(re_Pointfloat, re_Expfloat) + r"[fF]"
    re_Fixednumber = _group(re_Pointfloat, re_Expfloat) + r"[xX]"
    re_Special = _group(*(re.escape(op) for op in sorted(operators, key=len, reverse=True)))
    re_Quote = r"\\?" + _group('"""', "'''", '"', "'")
    re_Newline = r" *\r?\n"
    re_LineCont = r"\\\r?\n"
    re_InvalidLineCont = r"\\.+"


@dataclass(slots=True)
class TokenInfo:
    """
    Token container with position metadata.
    """
    type: TokenType
    value: str
    filename: str
    lineno: int
    column: int
    lexpos: int
    end_lineno: int = 0
    end_column: int = 0

    def __post_init__(self):
        self.end_lineno, self.end_column = self._get_endpos()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        token_range = f"{self.lineno},{self.column}-{self.end_lineno},{self.end_column}({self.lexpos})"
        return f"{token_range:<25}{self.type.name:<25}{self.value!r}"

    def _get_endpos(self) -> tuple[int, int]:
        if "\n" not in self.value:
            return self.lineno, self.column + len(self.value)

        line_offset = self.value.count("\n")
        last_newline = self.value.rfind("\n")
        return self.lineno + line_offset, len(self.value) - last_newline - 1


class TokenType(Enum):
    """
    Enumeration of token kinds emitted by the lexer.
    """

    NAME = auto()
    INT = auto()
    FLOAT = auto()
    FIXED = auto()
    OP = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    STRING = auto()
    STRING_START = auto()
    STRING_MIDDLE = auto()
    STRING_END = auto()
    COMMENT = auto()
    COMMENT_SEP = auto()
    ENDMARKER = auto()
    ERRORTOKEN = auto()


class Lexer:
    """
    The main lexer class.
    """

    class LexerState(Enum):
        REGULAR = 0  # Normal code parsing mode
        STRING = 1  # Inside a string literal
        INTERPOLATION = 2  # Inside an interpolation block

    @dataclass
    class BracketRecord:
        bracket: str  # The bracket character
        pos: int  # Position in the source code where bracket appears
        interp: bool  # Whether this bracket starts an interpolation block

    @dataclass
    class QuoteRecord:
        quote: str  # The quote character(s)
        pos: int  # Position in the source code where quote appears
        raw: bool  # Whether this is a raw string (\"...")


    def __init__(
        self,
        source: str,
        filename: str,
        errors: err.ErrorCollector,
        skip_comments=False,
        tabsize=4,
        dump_tokens=False,
    ):
        # Normalize the input text: ensure single trailing newline, expand tabs, remove formfeeds
        source = (source.rstrip("\n") + "\n").expandtabs(tabsize).replace("\f", " ")

        # Store source lines for accurate position tracking across multiple lines
        self.source_lines = source.split("\n")
        self.filename = filename
        self.skip_comments = skip_comments
        self.dump_tokens = dump_tokens
        self.linecount = source.count("\n") + 1

        self.source = source  # Raw source text
        self.errors = errors  # Error collector

        # Current scanning position tracking
        self._pos = 0  # Current character index in text
        self._lineno = 1  # Current line number (1-indexed)
        self._column = 0  # Current column number (0-indexed)

        # Next character cache (empty string indicates EOF)
        self._nextchar = self.source[0]

        # Internal stacks and buffers
        self._state_stack = [self.LexerState.REGULAR]  # Stack of lexer states
        self._indents = [0]  # Stack of indentation levels
        self._bracket_stack: list[Lexer.BracketRecord] = []  # Stack for bracket matching
        self._quote_stack: list[Lexer.QuoteRecord] = []  # Stack for nested quotes (inside interpolations)
        self._buffer: list[str] = []  # Buffer for accumulating string content

        self._last_token_was_newline = True
        self._line_continuation = False

        # Cache compiled regex patterns for performance
        self._cached_regex_patterns = {}

    def get_error_info(self, length: int, pos: int | None = None):
        # Calculate starting line and column numbers
        start_lineno, start_column = self._calculate_lineno_column(pos)
        pos = pos if pos is not None else self._pos

        # Calculate ending line and column numbers
        end_pos = pos + length
        end_lineno, end_column = self._calculate_lineno_column(end_pos)

        # Create error information with source location details
        return err.ErrorInfo(self.filename, start_lineno, start_column, self.source, end_lineno, end_column)

    def report(
        self,
        error: err.CompilerError,
    ) -> None:
        """Report an error/warning to the error collector"""
        self.errors.add(error)

    def _advance(self) -> bool:
        # Advance one character ahead

        text = self.source
        pos = self._pos

        if self._at_end():
            self._nextchar = ""
            return False
        
        # Move to next character and update position tracking
        self._nextchar = text[pos + 1]
        if text[pos] == "\n":
            # Newline resets column and increments line number
            self._lineno += 1
            self._column = 0
        else:
            # For regular characters, just increment column
            self._column += 1

        self._pos += 1
        return True

    def _match(self, regex: str) -> re.Match | None:
        if self._at_end():
            return None
        
        # Use cached regex pattern for performance
        if regex in self._cached_regex_patterns:
            pattern = self._cached_regex_patterns[regex]
        else:
            pattern = re.compile(regex)
            self._cached_regex_patterns[regex] = pattern

        # Attempt match at current position
        match = pattern.match(self.source, self._pos)
        if not match:
            return None
        
        # Advance the lexer for each matched character
        for _ in match.group():
            self._advance()
        return match

    def _accept(self, char: str) -> bool:
        # Check if the next character matches the expected character and advance if so
        if self._nextchar:
            if self._nextchar == char:
                self._advance()
                return True
        return False

    def _maketoken(self, type: TokenType, string: str, pos: int | None = None) -> TokenInfo:
        # Calculate line and column numbers for the token
        lineno, column = self._calculate_lineno_column(pos)
        pos = pos or self._pos

        # Calculate start position considering multi-line tokens
        newlines = string.count("\n")

        if newlines == 0:
            # Single line token
            spos = (lineno, column - len(string))
        else:
            # Multi-line token
            lines = string.split("\n")
            source_firstline_len = len(self.source_lines[lineno - newlines - 1])
            spos = (lineno - newlines, source_firstline_len - len(lines[0]))

        return TokenInfo(type, string, self.filename, *spos, pos - len(string))

    def _calculate_lineno_column(self, pos: int | None):
        if pos is None:
            return self._lineno, self._column

        # Clamp position to valid range
        if pos < 0:
            pos = 0
        if pos > len(self.source):
            pos = len(self.source)

        # Count newlines to determine line number
        before_pos = self.source[:pos]
        lineno = before_pos.count("\n") + 1

        # Find start of current line to calculate column
        last_newline = before_pos.rfind("\n")
        if last_newline == -1:
            column = pos
        else:
            line_start = last_newline + 1
            column = pos - line_start

        return lineno, column

    def __iter__(self) -> Generator[TokenInfo, None, None]:
        def _iter():
            # Start with placeholder tokens
            t1 = t2 = t3 = TokenInfo(TokenType.ERRORTOKEN, "", "", 0, 0, 0)

            # Iterate through raw tokens and coalesce string parts
            for token in self._tokenize():
                t1, t2, t3 = t2, t3, token

                if (
                    t1.type == TokenType.STRING_START
                    and t2.type == TokenType.STRING_MIDDLE
                    and t3.type == TokenType.STRING_END
                ):
                    # Complete string: START + MIDDLE + END -> SINGLE STRING
                    yield TokenInfo(
                        TokenType.STRING,
                        t1.value + t2.value + t3.value,
                        t1.filename, t1.lineno, t1.column, t1.lexpos
                    )
                elif (
                    t1.type == TokenType.STRING_START
                    and t2.type == TokenType.STRING_MIDDLE
                    and t3.type != TokenType.STRING_END
                ):
                    # Partial string: START + MIDDLE but no END yet
                    yield from [t1, t2, t3]
                elif t2.type == TokenType.STRING_START and t3.type == TokenType.STRING_MIDDLE:
                    # Skip duplicate emission when sequence overlaps
                    continue
                elif t2.type == TokenType.STRING_START and t3.type != TokenType.STRING_MIDDLE:
                    # Only START token ready
                    yield from [t2, t3]
                elif t3.type != TokenType.STRING_START:
                    # Regular token, yield as-is
                    yield t3

        # Dump all the tokens if verbose
        for token in _iter():
            yield token
            if self.dump_tokens:
                utils.print_info("-> " + str(token))

    def _current_state(self):
        # Get the current state of the lexer from the top of the state stack
        return self._state_stack[-1]

    def _at_end(self):
        # Return a bool value that indicates if the lexer has reached the end
        return self._pos == len(self.source) - 1

    def _get_buffer_and_clear(self) -> str:
        # Get the accumulated buffer content and clear the buffer
        buffer = self._buffer
        string = "".join(buffer)
        buffer.clear()
        return string

    def _push_bracket_record(self, bracket: str, interp: bool = False):
        # Push a bracket record onto the bracket stack for matching
        self._bracket_stack.append(self.BracketRecord(bracket, self._pos - len(bracket), interp))

    def _pop_bracket_record(self) -> Lexer.BracketRecord | None:
        # Pop a bracket record from the bracket stack
        if len(self._bracket_stack) > 0:
            return self._bracket_stack.pop()

    def _push_quote_record(self, quote: str):
        # Push a quote record onto the quote stack for nesting
        raw = quote.startswith("\\")  # Raw strings begin with backslash
        self._quote_stack.append(self.QuoteRecord(quote.lstrip("\\"), self._pos - len(quote), raw))

    def _pop_quote_record(self) -> Lexer.QuoteRecord | None:
        # Pop a quote record from the quote stack
        if len(self._quote_stack) > 0:
            return self._quote_stack.pop()

    def _get_quote_record(self) -> Lexer.QuoteRecord | None:
        # Get the current quote record from the top of the stack without popping
        if len(self._quote_stack) > 0:
            return self._quote_stack[-1]

    def _tokenize(self) -> Generator[TokenInfo, None, None]:
        # Prime newline matching so initial indentation is handled correctly
        self._match(LexerConfig.re_Newline)

        # Handle leading indentation at the start of file (unexpected indent)
        pos = self._pos
        if _match := self._match(LexerConfig.re_Whitespace):
            string = _match.group(0).lstrip()
            self.report(err.UnexpectedIndent(self.get_error_info(len(string), pos)))

        # Tokenization loop
        while not self._at_end():
            if (
                self._current_state() == self.LexerState.REGULAR
                or self._current_state() == self.LexerState.INTERPOLATION
            ):
                # Process regular code or interpolation content
                yield from self._handle_regular_content()
            elif self._current_state() == self.LexerState.STRING:
                # Process string literal content
                yield from self._handle_string_content()

        # Handle end-of-file cleanup
        yield from self._handle_eof()
    
    def _handle_regular_content(self) -> Generator[TokenInfo, None, None]:
        spos = self._pos
        self._line_continuation = False
        
        # Check for newline first - it affects indentation state
        if match := self._match(LexerConfig.re_Newline):
            string = match.group(0)
            yield from self._handle_newline(string)
            self._last_token_was_newline = True
            return

        # Try matching various token types in priority order
        if match := self._match(LexerConfig.re_Floatnumber):
            # Float literal
            string = match.group(0)
            yield self._maketoken(TokenType.FLOAT, string)

        elif match := self._match(LexerConfig.re_Fixednumber):
            # Fixed-point literal
            string = match.group(0)
            yield self._maketoken(TokenType.FIXED, string)

        elif match := self._match(LexerConfig.re_Intnumber):
            # Integer literal
            string = match.group(0)
            yield self._maketoken(TokenType.INT, string)

        elif match := self._match(LexerConfig.re_Comment):
            # Comment
            if self.skip_comments:
                # Every comment ends with a newline
                self._last_token_was_newline = True
                return

            if not self._last_token_was_newline:
                # Emit a special COMMENT_SEP token to separate the comment and rest of the code
                yield self._maketoken(TokenType.COMMENT_SEP, "", spos)
            string = match.group(0)
            yield self._maketoken(TokenType.COMMENT, string)

        elif match := self._match(LexerConfig.re_Quote):
            # Start of string literal
            quote = match.group(0)
            self._push_quote_record(quote)
            yield self._maketoken(TokenType.STRING_START, quote)
            self._state_stack.append(self.LexerState.STRING)

        elif match := self._match(LexerConfig.re_Name):
            # Identifier or keyword
            string = match.group(0)
            yield self._maketoken(TokenType.NAME, string)

        elif match := self._match(LexerConfig.re_Special):
            # Special character (operators, brackets, etc.)
            string = match.group(0)
            yield from self._handle_special(string)

        elif match := self._match(LexerConfig.re_InvalidLineCont):
            # Invalid sequence after a trailing backslash
            length = len(match.group(0)) - 1
            self.report(err.CharAfterLineContinuation(self.get_error_info(length, spos + 1)))

        elif self._match(LexerConfig.re_Whitespace):
            # Ignore whitespace
            pass
            
        elif self._match(LexerConfig.re_LineCont):
            # Line continuation marker
            self._line_continuation = True

        else:
            # Any other character is treated as an error token
            yield from self._handle_error()

        # Reset newline flag
        self._last_token_was_newline = False

    def _handle_interpolation(self) -> Generator[TokenInfo, None, None]:
        # Get any buffered string content before the interpolation
        buffer = self._get_buffer_and_clear()
        if self._current_state() == self.LexerState.STRING:
            # Emit the string content that came before the interpolation marker
            # Adjust column accounting for the two-character open marker '\{'
            yield self._maketoken(TokenType.STRING_MIDDLE, buffer, self._pos - 2)

        # Emit the interpolation marker as an operator
        yield self._maketoken(TokenType.OP, "\\{")
        # Push bracket record to track closure of interpolation block
        self._push_bracket_record("{", True)
        # Switch to INTERPOLATION state to parse the expression
        self._state_stack.append(self.LexerState.INTERPOLATION)

    def _handle_string_content(self) -> Generator[TokenInfo, None, None]:
        quote_rec = self._get_quote_record()
        assert quote_rec is not None  # Avoid the complaint from type checkers :D

        quote = quote_rec.quote
        if self._accept("\\"):
            # Backslash encountered: handle escape sequence
            yield from self._handle_escape_sequence(quote_rec)

        elif self._match(quote):
            # Closing quote encountered: emit string middle (content) and string end tokens
            buffer = self._get_buffer_and_clear()
            quote_len = len(quote)

            self._pop_quote_record()
            # Emit the string content and closing quote
            yield self._maketoken(TokenType.STRING_MIDDLE, buffer, self._pos - quote_len)
            yield self._maketoken(TokenType.STRING_END, quote)
            # Return to previous state (REGULAR or INTERPOLATION)
            self._state_stack.pop()

        elif self._match(LexerConfig.re_Newline):
            # Newline inside a string
            if len(quote) == 3:
                # Triple-quoted strings can contain newlines
                self._buffer.append("\n")
            else:
                # Single-line strings cannot contain newlines
                self.report(err.UnterminatedString(self.get_error_info(len(quote), quote_rec.pos)))

        else:
            # Regular character inside string content: buffer it
            self._buffer.append(self._nextchar)
            self._advance()

    def _handle_escape_sequence(self, quote_rec: Lexer.QuoteRecord) -> Generator[TokenInfo, None, None]:
        spos = self._pos - 1  # Position of the backslash

        if quote_rec.raw:
            # Raw strings: backslashes are taken literally
            self._buffer.append("\\")
            return
        if self._accept("{") and self._current_state() == self.LexerState.STRING:
            # Start interpolation sequence
            yield from self._handle_interpolation()
            return
        if self._match(
            LexerConfig.re_Newline
        ):  # continued line: backslash + newline => ignore both
            return
        if self._accept(quote_rec.quote):  # escaped quote (e.g. '\"' in "...")
            self._buffer.append(quote_rec.quote)
            return
        
        # Try simple escapes
        for ch, esc_ch in LexerConfig.simple_escapes.items():
            if self._accept(ch):
                self._buffer.append(esc_ch)
                return

        # Try hex/unicode escapes like \x00, \u0000, \U00000000
        for ch, length in LexerConfig.hex_escapes.items():
            if not self._accept(ch):
                continue

            match = self._match(f"[0-9a-fA-F]{{{length}}}")
            if match:
                charid = int(match.group(0), 16)

                # Validate Unicode code point range
                if charid >= 0x110000:
                    self.report(err.InvalidHexEscape(self.get_error_info(length + 2, spos), True))
                    self._buffer.append(f"\\{ch}" + match.group(0))
                else:
                    self._buffer.append(chr(int(match.group(0), 16)))
            else:
                # Invalid hex digits
                self.report(err.InvalidHexEscape(self.get_error_info(length + 2, spos), True))
                self._buffer.append(f"\\{ch}")
            return
        
        # Unknown escape sequence: add a warning and preserve backslash
        if self._at_end():
            self.report(err.UnexpectedEOF(self.get_error_info(0)))
        self.report(err.InvalidEscapeSequence(self.get_error_info(2, spos), self._nextchar, True))
        self._buffer.append("\\")

    def _handle_newline(self, string: str) -> Generator[TokenInfo, None, None]:
        # Inside brackets, newlines are insignificant
        if self._bracket_stack:
            return
        
        # Collapse any following blank lines into the newline token
        while newline := self._match(LexerConfig.re_Newline):
            string += newline.group(0)

        # Emit the NEWLINE token
        yield self._maketoken(TokenType.NEWLINE, string)
        # Handle indentation for the next line
        yield from self._handle_indentation()

    def _handle_indentation(self) -> Generator[TokenInfo, None, None]:
        indents = self._indents
        column = 0
        spos = self._pos

        # Count spaces for the new indentation level
        while True:
            if self._accept(" "):
                column += 1
            elif self._at_end():
                # EOF: nothing to emit
                return
            else:
                break

        # Check if indentation increased
        if column > indents[-1]:
            indents.append(column)
            yield self._maketoken(TokenType.INDENT, " " * column)

        # Check if indentation decreased (need DEDENT tokens)
        while column < indents[-1]:
            if column not in indents:
                # Indentation doesn't match any previous level
                self.report(err.MismatchedUnindent(self.get_error_info(column, spos)))
            indents.pop()
            yield self._maketoken(TokenType.DEDENT, "")

    def _handle_special(self, string: str) -> Generator[TokenInfo, None, None]:
        yield self._maketoken(TokenType.OP, string)

        # Check if character is a bracket (opening or closing)
        if string not in list(LexerConfig.brackets.keys()) + list(LexerConfig.brackets.values()):
            return
        
        if string in LexerConfig.brackets:  # Opening bracket
            self._push_bracket_record(string)
            return

        # Closing bracket
        bracket_rec = self._pop_bracket_record()
        if bracket_rec is None or string != LexerConfig.brackets[bracket_rec.bracket]:
            # Mismatched bracket error
            self.report(err.MismatchedBracket(self.get_error_info(1), string))
            return
        
        # Check if this closes an interpolation block
        is_interpolation = bracket_rec.interp

        # If closing a block that began an interpolation, return to STRING state
        if self._current_state() == self.LexerState.INTERPOLATION and string == "}" and is_interpolation:
            self._state_stack.pop()

    def _handle_eof(self) -> Generator[TokenInfo, None, None]:
        # Check for line continuation
        if self._line_continuation:
            self.report(err.UnexpectedEOF(self.get_error_info(0)))
        
        # Check for unclosed strings
        if self._current_state() in [self.LexerState.INTERPOLATION, self.LexerState.STRING]:
            for quote_rec in self._quote_stack:
                quote = quote_rec.quote
                pos = quote_rec.pos
                self.report(err.UnterminatedString(self.get_error_info(len(quote), pos)))

        # Check for unclosed brackets
        if self._bracket_stack:
            for bracket_rec in self._bracket_stack:
                bracket = bracket_rec.bracket
                pos = bracket_rec.pos
                self.report(err.UnclosedBracket(self.get_error_info(len(bracket), pos), bracket))

        # Ensure file ends with a newline token
        if not self._last_token_was_newline:
            yield self._maketoken(TokenType.NEWLINE, "", self._pos)

        # Emit DEDENT tokens to close all open indentation levels
        indents = len(self._indents) - 1
        if indents > 0:
            for _ in range(indents):
                yield self._maketoken(TokenType.DEDENT, "")

        # Final marker indicating end of token stream
        yield self._maketoken(TokenType.ENDMARKER, "")

    def _handle_error(self) -> Generator[TokenInfo, None, None]:
        # Handle unknown or invalid characters
        error_char = self._nextchar
        self.report(err.InvalidCharacter(self.get_error_info(1), error_char))
        self._advance()
        yield self._maketoken(TokenType.ERRORTOKEN, error_char)


if __name__ == "__main__":
    from vcs.cli import cli
    cli()
