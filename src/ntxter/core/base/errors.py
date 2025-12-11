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