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

def test_add_raises_typeerror_when_a_invalid():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add("1", 2)

def test_add_raises_typeerror_when_b_invalid():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add(1, "2")

def test_add_raises_typeerror_when_a_invalid():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add("1", 2)

def test_add_raises_typeerror_when_b_invalid():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add(1, "2")

def disabled_test_x_add_raises_typeerror_when_a_invalid():
    import pytest
    from example_pkg.math_utils import x_add

    with pytest.raises(TypeError):
        x_add("1", 2)

def disabled_test_x_add_raises_typeerror_when_b_invalid():
    import pytest
    from example_pkg.math_utils import x_add

    with pytest.raises(TypeError):
        x_add(1, "2")

class _AddableButInvalid:
    """Not an int/float, but supports + so mutants can't 'accidentally' pass."""
    def __add__(self, other):  # pragma: no cover
        return 123

    def __radd__(self, other):  # pragma: no cover
        return 123


def test_add_rejects_invalid_a_even_if_addition_would_work():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add(_AddableButInvalid(), 2)


def test_add_rejects_invalid_b_even_if_addition_would_work():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError):
        add(2, _AddableButInvalid())

def test_add_typeerror_message_is_stable_for_invalid_args():
    import pytest
    from example_pkg.math_utils import add

    with pytest.raises(TypeError, match=r"^add\(\) expects int or float arguments$"):
        add(object(), 1)
