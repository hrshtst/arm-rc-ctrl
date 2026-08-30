"""Planted type error: returning an int where a str is declared."""

from __future__ import annotations


def wrong_return(value: int) -> str:
    """Return the wrong type on purpose."""
    return value
