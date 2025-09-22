class DuplicateKeyError(KeyError):
    def __init__(self, message="Duplicated key", error_code=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code else -1

    def __str__(self):
        return f"DuplicateKeyError: {self.message} (Code: {self.error_code})"