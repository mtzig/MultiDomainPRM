import torch
from transformers import (
    StoppingCriteria,
)
from typing import List
import re

def condense_newlines(text):
    """
    Condenses multiple consecutive "\n\n" patterns in the input string to a single "\n\n".

    Args:
        s (str): The input string to be processed.

    Returns:
        str: The processed string with condensed newlines.
    """
    # Use a regular expression to match two or more consecutive "\n\n" patterns,
    # possibly separated by whitespace characters.
    # Replace the matched patterns with a single "\n\n".
    # The pattern breakdown:
    # - (?:\n\n\s*) is a non-capturing group that matches "\n\n" followed by any whitespace.
    # - {2,} specifies that the non-capturing group should repeat two or more times.
    collapsed_text = re.sub(r'\n{2,}', '\n\n', text)
    collapsed_text = re.sub(r'(?:\n\n\s*){2,}', '\n\n', collapsed_text)
    return collapsed_text


class MultiEosStoppingCriteria(StoppingCriteria):
    """
    Custom StoppingCriteria to check if any of multiple end-of-sequence tokens have been generated.
    """

    def __init__(self, eos_token_ids: List[int]) -> None:
        """
        Initializes the MultiEosStoppingCriteria.

        Args:
            eos_token_ids (List[int]): A list of end-of-sequence token IDs.
        """
        super().__init__()
        self.eos_token_ids: List[int] = eos_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: dict
    ) -> bool:
        """
        Determines whether generation should stop based on the last generated token.

        Args:
            input_ids (torch.LongTensor): The current sequence of generated tokens, shape [batch_size, sequence_length].
            scores (torch.FloatTensor): The logits for the next token, shape [batch_size, vocab_size].
            **kwargs: Additional optional arguments.

        Returns:
            bool: True if the last token is one of the specified end-of-sequence tokens; otherwise, False.
        """
        # Assuming batch_size is 1
        last_token: int = input_ids[0, -1].item()
        return last_token in self.eos_token_ids
    
if __name__ == "__main__":
    original_text = 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nIn additon to any other constraints, please solve the problem using a specific format. Each reasoning step should be separated by a blank line, and the final answer should be given in the format \\boxed{<answer>}.\n\n\n\n# Solution\n\n## Step 1: Recall the conversion formulas from rectangular coordinates to polar coordinates.\nThe conversion formulas from rectangular coordinates $(x,y)$ to polar coordinates $(r,\\theta)$ are given by $r = \\sqrt{x^2 + y^2}$ for the radial coordinate and $\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)$ for the angular coordinate \n\n## Step 2: Apply the formula for the radial coordinate $r$.\nGiven the point $(0,3)$, we substitute $x = 0$ and $y = 3$ into the formula for $r$. This gives us $r = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3$ \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n'
    original_text = 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nIn additon to any other constraints, please solve the problem using a specific format. Each reasoning step should be separated by a blank line, and the final answer should be given in the format \\boxed{<answer>}.\n\n# Solution\n\n## Step 1: Recall the conversion formulas from rectangular coordinates to polar coordinates.\nThe conversion formulas from rectangular coordinates $(x,y)$ to polar coordinates $(r,\\theta)$ are given by $r = \\sqrt{x^2 + y^2}$ and $\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)$ \n\n## Step 2: Calculate the value of $r$ using the formula $r = \\sqrt{x^2 + y^2}$.\n$r = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3$\n\n\n## Step 3: Calculate the value of $\\theta$ using the formula $\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)$.\n$\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right) = \\tan^{-1}(\\infty) = \\frac{\\pi}{2}$\n\n\n## Step 4: Combine the values of $r$ and $\\theta$ to obtain the polar coordinates.\nThe polar coordinates are $\\left(3, \\frac{\\pi}{2}\\right)$\n\n\nThe final answer is: $\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}$ \n\n'
    print("Original Text:")
    print(repr(original_text))
    
    condensed_text = condense_newlines(original_text)
    print("\nCondensed Text:")
    print(repr(condensed_text))

