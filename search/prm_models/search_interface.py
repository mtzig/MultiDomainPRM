from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchOutput:
    answer: Optional[str] = None
    output: Optional[str] = None
    alternate_outputs: Optional[list[str]] = None


class Search(ABC):
    """An interface for LLM search algorithm."""

    @abstractmethod
    def __call__(self, problem: str) -> SearchOutput:
        pass
