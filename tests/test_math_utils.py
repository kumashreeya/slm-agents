import pytest

from example_pkg.math_utils import add


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 2, 3),
        (1.5, 2.5, 4.0),
        (-1, 5, 4),
    ],
)
def test_add_happy_path(a, b, expected):
    assert add(a, b) == expected


@pytest.mark.parametrize("a,b", [("1", 2), (1, "2"), (None, 2)])
def test_add_type_errors(a, b):
    with pytest.raises(TypeError):
        add(a, b)
