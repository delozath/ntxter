from tomlkit import item
from torch import isin
import yaml
from typing import Any
from functools import reduce

from ntxter.validation import SingleAssignWithType
from ntxter.data.dtypes import NestedDictionary

class ProjectConfigLoader:
    schema = SingleAssignWithType(list)

    def load(self, pfname):
        schema = list()
        with open(pfname) as load:
            for read in yaml.safe_load_all(load):
                key, item = [*read.items()][0]
                setattr(self, key, item)
                schema.append(key)
        #
        if hasattr(self, 'general'):
            setattr(self, 'project', self.general.get('project', 'unnamed project'))
            setattr(self, 'stage'  , self.general.get('stage', 'void'))
            self.schema = ['project', 'stage'] + schema
        else:
            raise ValueError("YAML file structure failed to include a 'general' document")

class ProjectCfgNavigator:
    def __init__(self, cfg):
        self._cfg = cfg
        for c in cfg.schema:
            item = getattr(cfg, c)
            if isinstance(item, dict):
                setattr(self, c, NestedDictionary(item))
            else:
                setattr(self, c, item)
        #TODO: reference for avoid duplicating dictionaries
        self.stage = (
            NestedDictionary(getattr(cfg, cfg.stage)) 
            if hasattr(cfg, cfg.stage) 
            else 'empty'
         )
        self.stage['store', '1M', 'paths']
        breakpoint()
    
