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
    schema = SingleAssignWithType(list)

    def __init__(self, cfg):
        _ = self._integrity_header(cfg)
        self._cfg = cfg
        self.schema = cfg.schema

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
    #
    #TODO: full integrity header check
    def _integrity_header(self, cfg):
        try:
            gral = cfg.general
        except:
            raise AttributeError("'general' is a mandatory attribute within cfg to continue")
        else:
            stage = gral.get('stage', '')
            if stage in cfg.schema:
                return
            else:
                raise AttributeError(f"Stage '{stage}' is not part of the available schema in cfg object")

    def __repr__(self) -> str:
        return f"{type(self).__name__} scheme: {self._cfg.schema}"
    
