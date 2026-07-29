from typing import override
from dataclasses import dataclass

import hmac
import hashlib

from ntxter.core.base.errors import HashingError
from ntxter.core.base.classes import BaseCallableClass


@dataclass
class BaseAnonimizer(BaseCallableClass):
    _key: str | bytes | bytearray

    @property
    def key(self):
        print("Data not readable")

    @key.setter
    def key(self, value):
        print("Data not modifiable")

    @key.deleter
    def key(self):
        print("Data not modifiable")

    @override
    def __call__(self, *args, **kwargs) -> str:
        hashed = self._anonimization(*args, **kwargs)
        if hashed is None:
            raise HashingError(f"hashing process return NoneType in class {type(self).__name__}")
        return hashed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._key = None
        self.__dict__.clear()
        return False
    
    def _anonimization(self, *args, **kwargs) -> str | None:
        return


class SecureHasherFromList(BaseAnonimizer):
    def __post_init__(self):
        if not isinstance(self._key, (bytes, bytearray)):
            raise ValueError(f"`key` value must be bytes | bytearra; `{type(self._key)}` type provided.")

    @override
    def _anonimization(self, elements: list, *args, **kwargs) -> str:
        if not isinstance(elements, list):
            raise ValueError(f"`elements` value must be list; `{type(elements)}` type provided.")
        
        payload = (
            "|".join([f"{e}" for e in elements])
               .encode('utf-8')
            )
        
        hashed = hmac.new(
            key=self._key,
            msg=payload,
            digestmod=hashlib.sha256
        )
        
        return hashed.hexdigest()

    def __str__(self):
        return "SecureHasherFromList"

    def __repr__(self):
        return "SecureHasherFromList"
