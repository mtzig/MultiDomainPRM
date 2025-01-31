import math
import statistics
from typing import Optional
from tqdm import tqdm
import torch
from .prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers import MistralModel, MistralPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores

class ReasonEval7bPRM(PRM):
    def __init__(
        self,
        aggregation: str = "full",#the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )


        self.tokenizer = AutoTokenizer.from_pretrained('GAIR/ReasonEval-7B')
        self.model = ReasonEval_7B.from_pretrained('GAIR/ReasonEval-7B').eval().to(self.device)
        self.aggregation = aggregation

      

    def __call_single(self, single_beam: str) -> float | list[float]:


        single_beam_split = single_beam.split('\n\n')
        # print("single beam split:")
        # print(len(single_beam_split))
        question = single_beam_split[0]
        reasoning_steps = single_beam_split[1:]
        # for index, each in enumerate(single_beam_split):
        #     print(index)
        #     print(each)

        # Preprocessing input
        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        step_separator = f"{self.tokenizer.pad_token}"
        combined_steps = ""
        for steps in reasoning_steps:
            combined_steps += steps + step_separator
        prompt = PROMPT_FORMAT.format(input=question)
        tokenized_result = self.tokenizer(prompt + step_separator + combined_steps)['input_ids']

        ## Separating labels and adjusting token IDs
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]
        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)

        adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)

        assert len(labeled_token_indices) == len(reasoning_steps)

        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        # Evaluating reasoning steps using ReasonEval
        with torch.no_grad():
            reasoning_scores = self.model(adjusted_token_ids.to(self.device), attention_mask.to(self.device))[0,labeled_token_indices , :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()


        ## S_{validity} = p_{neutral} + p_{positive}
        step_probs =  [(score[1] + score[2]) for score in scores]

        if self.aggregation == "min":
            return min(step_probs)
        elif self.aggregation == "max":
            return max(step_probs)
        elif self.aggregation == "mean":
            return statistics.mean(step_probs)
        elif self.aggregation == "prod":
            return math.prod(step_probs)
        elif self.aggregation == "last":
            return step_probs[-1]
        elif self.aggregation == "full":
            return step_probs
        else:
            raise NotImplementedError

    def __call__(self, steps: list[str]) -> list[StepScore]:
        """
        Args:
            steps (list[str]): A list of reasoning solutions.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """

        result = []

        for beam in steps:#each beam is a cot solution, each step_score is a list of prm step scores for this solution(if aggregation methods is full)
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result


