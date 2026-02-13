import re


def validate_email(email: str) -> bool:
    """
    Validates an email address.

    Args:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))
