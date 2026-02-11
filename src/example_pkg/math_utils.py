from __future__ import annotations

Number = int | float


def add(a: Number, b: Number) -> Number:
    """Return the sum of two numbers.

    Raises:
        TypeError: if a or b is not int/float
    """
    if not isinstance(a, int | float) or not isinstance(b, int | float):
        raise TypeError("add() expects int or float arguments")
    return a + b
