import pytest

from example_pkg.math_utils import add


class _AddableButInvalid:
    """Object that *can* be added by Python, but should be rejected by add()."""

    def __add__(self, other):
        return 123

    def __radd__(self, other):
        return 123


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (1.5, 2.5, 4.0),
        (-1, 5, 4),
        (0, 0, 0),
        (10, -3, 7),
    ],
)
def test_add_happy_path(a, b, expected):
    assert add(a, b) == expected


@pytest.mark.parametrize(
    "a,b",
    [
        ("a", 2),
        (1, "b"),
        (None, 2),
        (2, None),
        ([], 1),
        ({}, 1),
        (object(), 1),
    ],
)
def test_add_type_errors_have_clear_message(a, b):
    # Stronger than just "raises TypeError": also checks the message.
    with pytest.raises(TypeError, match=r"^add\(\) expects int or float arguments$"):
        add(a, b)


def test_add_rejects_invalid_addable_even_if_python_addition_would_work():
    weird = _AddableButInvalid()

    # Prove Python itself would allow this addition (so if type-check is mutated away,
    # add() might incorrectly return 123 and this test will fail).
    assert weird + 2 == 123
    assert 2 + weird == 123

    with pytest.raises(TypeError, match=r"^add\(\) expects int or float arguments$"):
        add(weird, 2)

    with pytest.raises(TypeError, match=r"^add\(\) expects int or float arguments$"):
        add(2, weird)
