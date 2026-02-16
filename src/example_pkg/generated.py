import re


def validate_email(email: str) -> bool:
    """
    Validates an email address using regex.

    Args:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email is valid, False otherwise.
    """

    # Regular expression pattern for validating email addresses
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    # Check if the email matches the pattern
    return bool(re.match(pattern, email))
