import re


class EmailValidator:
    """A class to validate email addresses."""

    def __init__(self):
        """Initialize the validator with a regular expression pattern."""
        self.pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    def validate_email(self, email: str) -> bool:
        """
        Validates an email address.

        Args:
            email (str): The email address to be validated.

        Returns:
            bool: True if the email is valid, False otherwise.
        """
        return bool(re.match(self.pattern, email))


def validate_email(email: str) -> bool:
    """A function to validate an email address."""
    validator = EmailValidator()
    return validator.validate_email(email)
