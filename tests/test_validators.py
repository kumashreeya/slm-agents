from example_pkg.validators import validate_email


def test_validate_email_valid():
    """Test valid email addresses."""
    assert validate_email("test@example.com")
    assert validate_email("user.name@example.co.uk")


def test_validate_email_invalid():
    """Test invalid email addresses."""
    assert not validate_email("invalid-email")
    assert not validate_email("@example.com")
    assert not validate_email("test@")
