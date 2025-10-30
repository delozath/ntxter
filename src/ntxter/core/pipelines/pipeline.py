from abc import ABC, abstractmethod


from sklearn.pipeline import Pipeline


from ntxter.core.base.descriptors import SetterAndGetter


class BasePipeline():
    pipeline_ =  SetterAndGetter()

    def insert_stage(self, place: int, stage: tuple):
        self._pipeline_.steps.insert(place, stage)