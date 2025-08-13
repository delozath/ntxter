import yaml

class ProjectConfigLoader:
    def load(self, pfname):
        with open(pfname) as load:
            for read in yaml.safe_load_all(load):
                key, item = [*read.items()][0]
                setattr(self, key, item)
        #
        try:
            setattr(self, 'project', self.general['project'])
            setattr(self, 'stage'  , self.general['stage'])
        except:
            raise AttributeError("The key:general must to be set in the yaml file")