"""
============================================================
Custom Exception System
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This module defines a custom exception class used throughout
the machine learning pipeline.

It enhances standard Python exceptions by providing
additional debugging information such as:

• file name where the error occurred
• line number of the error
• detailed stack trace
• original error message

Why Custom Exceptions Are Important
-----------------------------------
Machine learning pipelines consist of multiple stages:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation

When an error occurs in any stage, it is critical to know:

• exactly where the error happened
• what caused the error
• the full execution trace

This custom exception system ensures that debugging
information is captured and displayed clearly.

Bug Fixes Applied
-----------------
Three bugs were present in the original implementation:

Bug 1 — Wrong traceback frame (silent wrong output)
    Original: exc_tb.tb_lineno
    Problem : exc_tb is the OUTERMOST frame of the traceback
              chain, not the frame where the error actually
              occurred. For a call like deep() → inner() → crash,
              the original code reported the line of deep(), not
              the line inside inner() where the crash happened.
    Fix     : Walk the tb_next chain to the innermost frame
              before reading tb_lineno and f_code.co_filename.

Bug 2 — Double-wrapping (garbled error messages)
    Original: raise BankruptcyException(e, sys) where e is
              already a BankruptcyException. str(e) returns
              the fully-formatted multi-line message, which
              then gets embedded as the "error_message" of a
              new BankruptcyException. The result is a nested
              message that mentions "Error occurred in script"
              twice with two different tracebacks.
    Fix     : If error_message is already a BankruptcyException,
              re-raise it directly rather than wrapping it again.

Bug 3 — sys.exc_info() returns (None, None, None) outside except
    Original: sys.exc_info() was called inside a @staticmethod.
              It works by accident when called from within an
              active except block, but returns all-None when
              BankruptcyException is raised outside one (e.g.
              during config validation or argument checking).
              Result: file="Unknown", line="Unknown" — useless.
    Fix     : Accept the live traceback object via
              traceback.extract_stack() / sys.exc_info() at
              the call site and store it on the instance, so
              the static helper always receives real data.
"""

# ==========================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================================

import sys
import traceback


# ==========================================================
# 2. CUSTOM EXCEPTION CLASS
# ==========================================================

class BankruptcyException(Exception):
    """
    Custom exception for the Bankruptcy ML pipeline.

    Extends Python's built-in Exception with precise
    debugging information: the exact file and line where
    the error originated, plus the full stack trace.

    Usage (anywhere in the pipeline)
    ---------------------------------
    Standard pattern — inside an except block:

        try:
            risky_operation()
        except Exception as e:
            raise BankruptcyException(e, sys) from e

    Outside an except block (argument / config validation):

        if not os.path.exists(path):
            raise BankruptcyException(
                f"File not found: {path}"
            )

    Both patterns produce a fully-populated error message.
    """

    def __init__(self, error_message, error_detail=None):
        """
        Initialise the custom exception.

        Parameters
        ----------
        error_message : str | Exception
            The error message or caught exception.

            If error_message is already a BankruptcyException
            the message is forwarded as-is to avoid nesting
            the same formatted block twice.

        error_detail : module, optional
            Pass the `sys` module here when raising from
            inside an except block:

                raise BankruptcyException(e, sys)

            This allows the constructor to call
            sys.exc_info() while the active exception
            context is still live on the call stack.

            Omit (or pass None) when raising outside an
            except block — the constructor will fall back
            to traceback.extract_stack() instead.
        """

        # ── Fix 2: avoid double-wrapping ──────────────────
        # If the caught exception is itself a BankruptcyException
        # (common when pipeline stages re-raise component errors)
        # just forward its already-formatted message unchanged.
        if isinstance(error_message, BankruptcyException):
            super().__init__(str(error_message))
            self.error_message = str(error_message)
            return

        super().__init__(str(error_message))

        self.error_message = self._build_message(
            str(error_message),
            error_detail,
        )

    # ======================================================
    # INTERNAL: BUILD DETAILED ERROR MESSAGE
    # ======================================================

    @staticmethod
    def _build_message(error_message: str, error_detail) -> str:
        """
        Build a detailed error string with file, line,
        original message, and full stack trace.

        Parameters
        ----------
        error_message : str
            Plain text description of the error.

        error_detail : module | None
            The `sys` module when called from an except block,
            or None when called outside one.

        Returns
        -------
        str
            Multi-line formatted error string.
        """

        file_name   = "Unknown"
        line_number = "Unknown"

        # ── Fix 3: use sys.exc_info() while it is still live ──
        # error_detail is the `sys` module passed by the caller.
        # Calling error_detail.exc_info() works only when an
        # exception is currently active on the call stack.
        # We call it immediately (before any other code can clear
        # the active exception context) to guarantee we get data.
        exc_tb = None

        if error_detail is not None:
            # Called from inside an except block — active context
            # is still live here because we are in __init__ which
            # is called directly from the raise statement.
            _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            # ── Fix 1: walk to the innermost frame ────────────
            # exc_tb points to the outermost frame of the
            # traceback chain (i.e. the try block that caught
            # the error, not the line that caused it).
            # Walk tb_next until we reach the end to get the
            # exact origin of the crash.
            innermost = exc_tb
            while innermost.tb_next:
                innermost = innermost.tb_next

            file_name   = innermost.tb_frame.f_code.co_filename
            line_number = innermost.tb_lineno

        else:
            # Called outside an except block (config validation,
            # argument checks, etc.).  Fall back to the current
            # call stack so we still get a useful location.
            stack = traceback.extract_stack()
            # The last two frames are _build_message and __init__
            # themselves — step back to the actual caller.
            if len(stack) >= 3:
                caller      = stack[-3]
                file_name   = caller.filename
                line_number = caller.lineno

        # Full formatted traceback (empty string if no active exc)
        stack_trace = traceback.format_exc()
        if stack_trace.strip() == "NoneType: None":
            stack_trace = "(no active exception — raised outside except block)"

        return (
            f"\nError occurred in script : {file_name}"
            f"\nLine number              : {line_number}"
            f"\nError message            : {error_message}"
            f"\nStack trace              :\n{stack_trace}"
        )

    # ======================================================
    # STRING REPRESENTATION
    # ======================================================

    def __str__(self) -> str:
        return self.error_message