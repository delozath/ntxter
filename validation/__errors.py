class _StringMessage:
    def __str__(self):
        return f"{type(self).__name__}: {self.message} (Code: {self.error_code})"


class DuplicateKeyError(KeyError, _StringMessage):
    def __init__(self, message="Duplicated key", error_code=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code else -1


class UnsetAttributeError(ValueError, _StringMessage):
    def __init__(self, message="Attribute has not been assigned yet", error_code=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code else -1


class ExpectedTypeError(TypeError, _StringMessage):
    def __init__(self, expected: str, received: str, error_code: int | None = None):
        message = f"type expected ({expected}), but ({received}) was provided"
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code else -1