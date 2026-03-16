"""
============================================================
Exception Handling Tests
Bankruptcy Risk Intelligence System
============================================================

Purpose
-------
This test module verifies that the custom BankruptcyException
class correctly captures and reports detailed debugging
information when errors occur.

Why This Test Matters
---------------------
Custom exceptions are used throughout the ML pipeline to
provide detailed debugging information.

When an error occurs, the exception should capture:

• script file name
• line number of the error
• original error message
• full stack trace

This test ensures that the exception system behaves as
expected.
"""

# ==========================================================
# IMPORT REQUIRED MODULES
# ==========================================================

import sys
import pytest

from bankruptcy.utils.exception import BankruptcyException


# ==========================================================
# TEST: CUSTOM EXCEPTION MESSAGE
# ==========================================================

def test_bankruptcy_exception_contains_detailed_message():
    """
    Verify that BankruptcyException contains detailed
    debugging information when raised.
    """

    try:

        # ----------------------------------------------
        # Trigger an intentional error
        # ----------------------------------------------

        x = 1 / 0

    except Exception as e:

        exc = BankruptcyException(e, sys)


    # ==================================================
    # 1️⃣ VERIFY EXCEPTION TYPE
    # ==================================================

    assert isinstance(exc, BankruptcyException)


    message = str(exc)


    # ==================================================
    # 2️⃣ VERIFY MESSAGE CONTAINS DEBUG INFORMATION
    # ==================================================

    assert "Error occurred in script" in message

    assert "Line number" in message

    assert "division by zero" in message


    # ==================================================
    # 3️⃣ VERIFY STACK TRACE EXISTS
    # ==================================================

    assert "Traceback" in message