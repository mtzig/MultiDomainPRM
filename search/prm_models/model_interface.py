from abc import ABC, abstractmethod
from typing import Any, Optional

from transformers import (  # type: ignore
    PreTrainedModel,
    PreTrainedTokenizer,
)


class BaseModel(ABC):
    """This is an abstract base class for models to be used with the Search class."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        end_step_str: list[str],
        end_step_id: int | list[int],
        end_solution_str: str,
        end_solution_id: int,
        meta_prompt: str,
        device: str,
    ) -> None:
        """
        Initialize the model with the required components.

        Args:
            model (PreTrainedModel): The HuggingFace model used for inference.
            tokenizer (PreTrainedTokenizer): Tokenizer for converting text to tokens and vice versa.
            end_step_str (list[str]): Strings form of the reasoning step complete tokens.
            end_step_id (list[int]): Token IDs marking the end of a reasoning step.
            end_solution_str (str): String form of the solution complete token.
            end_solution_id (int): Token ID marking the end of the solution.
            meta_prompt (str): Additional prompt that serves as context to guide the reasoning process.
            device (str): Model device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.end_step_id = end_step_id
        self.end_step_str = end_step_str
        self.end_solution_str = end_solution_str
        self.end_solution_id = end_solution_id
        self.meta_prompt = meta_prompt
        self.device = device

    @abstractmethod
    def add_meta_prompt(self, prompt: str) -> str:
        """
        Method to integrate the meta prompt into the given prompt.

        The meta prompt serves as additional context or guidance for the model to
        help structure the reasoning process. This method should be implemented by subclasses
        to customize how the meta prompt is combined with the input prompt.

        Args:
            prompt (str): The initial prompt provided to the model.

        Returns:
            str: The modified prompt with the meta prompt included.
        """
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Method to extract the final answer from the model's output text.

        This method parses the provided text to locate the answer to the problem, typically
        by looking for specific patterns like `/boxed{answer}`. This method should be
        customized in subclasses depending on the answer extraction logic needed.

        Args:
            text (str): The raw text output from the model.

        Returns:
            Optional[str]: The extracted answer from the text, if an answer is present.
        """
        pass

    @abstractmethod
    def get_delimeter_ids(self) -> list[int]:
        """
        Returns a list of the token IDs for the reasoning step complete delimiter and solution complete delimiter.

        Returns:
            list[int]: A tuple containing the reasoning step complete ID and solution complete ID.
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: dict[str, Any] = {},
        add_meta_prompt: bool = False,
    ) -> list[str]:
        """Returns the decoded outputs and of prompts."""
        pass

    @abstractmethod
    def is_complete(self, text: str) -> bool:
        pass
