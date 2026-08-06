class DatabaseConnectionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class RegistryError(Exception):
    def __init__(self, message: str):
        print(f"Registry Error: data already exists.")
        super().__init__(message)

class UnknownDataTypeError(Exception):
    def __init__(self, dtype: str):
        self.dtype = dtype
        self.message = f"Unsupported data type: `{dtype}`."
        super().__init__(self.message)

class HashingError(Exception):
    def __init__(self, message=None) -> None:
        extra_info = "" if message is None else f": {message}"
        self.message = f"Unexpected error while hashing{extra_info}"
        super().__init__(self.message)