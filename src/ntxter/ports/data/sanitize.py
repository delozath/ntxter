from ast import pattern
from typing import Dict, Generator, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StringSanitization:
    char_map: Dict[str, str]
    str_map: Dict[str, str] | None = None
    pattern: Dict[str, str] | None = None
    whitespace: str = "_"
    force_case: Literal["lower", "upper", "keep"] = "lower"

    def __post_init__(self):
        if self.force_case not in ["lower", "upper", "keep"]:
            self.force_case = "lower"
            print("Warning: `force_case` option not recognized, defaulting to 'lower'.")
        if not (isinstance(self.pattern, dict) or self.pattern is None):
            raise ValueError("`pattern` must be a dict or None.")
        if not (isinstance(self.str_map, dict) or self.str_map is None):
            raise ValueError("`str_map` must be a dict or None.")


class DataSanitizer[T, U](ABC):
    config: U
    
    @abstractmethod
    def sanitize(
        self,
        data: T,
      ) -> T | Generator | None:
        ...