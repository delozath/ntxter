from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class Orchestrator(ABC):
    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError("Subclasses must implement the 'execute' method.")

