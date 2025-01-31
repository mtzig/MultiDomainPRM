from torch import Tensor
from transformers import PreTrainedTokenizer, StoppingCriteria


class BoxStoppingCriteria(StoppingCriteria):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, 
        eos_token_ids: list[int],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids

    def __call__(self, output_ids_batch: Tensor, scores: Tensor, **kwargs: dict) -> bool:
        return all(self.is_complete(output_ids) for output_ids in output_ids_batch)

    def is_complete(self, output_ids: Tensor) -> bool:
        if output_ids[-1] in self.eos_token_ids:
            return True
        
        text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        i = text.rfind("\\boxed{") + len("\\boxed{")
        if i == -1:
            return False

        j = i
        brace_count = 1
        while brace_count > 0 and j < len(text) - 1:
            j += 1
            if text[j] == "{":
                brace_count += 1
            elif text[j] == "}":
                brace_count -= 1
        
        return brace_count == 0 and text[i:j] != "<answer>"
