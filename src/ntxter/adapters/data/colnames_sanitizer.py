import re
from typing import Generator

import pandas as pd

from functools import reduce


from ntxter.ports.data.sanitize import DataSanitizer, StringSanitization


class ColnamesSanitizer(DataSanitizer[str | list[str], StringSanitization]):
    #NOTE: remove (), {}, []
    config: StringSanitization

    def __init__(
        self, 
        char_map,
        str_map=None,
        pattern=None,
        whitespace="_",
        force_case=""
     ) -> None:
            self.config = StringSanitization(
                char_map=char_map,
                str_map=str_map,
                pattern=pattern,
                whitespace=whitespace,
                force_case=force_case
            )
    
    def sanitize(
        self,
        data: str | list[str],
      ) -> str | list[str] | Generator:
        mapping = self.config.char_map | {' ': self.config.whitespace}
        for col in data:
            new = self.char_replace(col, mapping)
            new = self.str_replace(new, self.config.str_map)
            new = self.to_case(new)
            new = self.regex_sub(new, self.config.pattern)
            
            yield col, new
    
    def char_replace(self, s: str, mapping) -> str:
        return "".join(mapping.get(ch, ch) for ch in s)
    
    def str_replace(self, s: str, mapping) -> str:
        if mapping is None:
            return s
        return self.char_replace(s, mapping)
    
    def to_case(self, string: str) -> str:
        if self.config.force_case == "lower":
            return string.lower()
        elif self.config.force_case == "upper":
            return string.upper()
        return string
    
    def regex_sub(
        self,
        s: str,
        patterns: dict[str, str] | None,
      ) -> str:
        if patterns is None:
            return s
        pre_compiled = [(re.compile(k), i) for k, i in patterns.items()]
        
        s = reduce(lambda acc, pr: pr[0].sub(pr[1], acc), pre_compiled, s)
        return s

class PandasFrameColnamesSanitizer:
    def __init__(
        self, 
        char_map,
        str_map=None,
        pattern=None,
        whitespace="_",
        force_case=""
     ):
       self.sanitizer = ColnamesSanitizer(
            char_map=char_map,
            str_map=str_map,
            pattern=pattern,
            whitespace=whitespace,
            force_case=force_case
            )

    def run(
        self,
        data: pd.DataFrame,
      ) -> tuple[pd.DataFrame, pd.DataFrame]:
        colnames = data.columns.tolist()
        newcolnames = {old: new for old, new in self.sanitizer.sanitize(colnames)}
        if len(set(newcolnames.values())) != len(newcolnames.values()):
            raise ValueError("Sanitization produced duplicated column names. Future implementation to conflict resolution.")
        
        data.rename(columns=newcolnames, inplace=True)
        newcolnames = pd.DataFrame([newcolnames]).T.reset_index()
        newcolnames.columns = ['original', 'sanitized']
        return data, newcolnames