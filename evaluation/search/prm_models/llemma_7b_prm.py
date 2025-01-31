import math
import statistics
from typing import Optional

import torch
from .prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils import condense_newlines


class Llemma7bPRM(PRM):
    def __init__(
        self,
        aggregation: str = "min",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(self.device)
        # input()
        self.model = AutoModelForCausalLM.from_pretrained(
            "ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf",
            quantization_config=quantization_config,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.begin_solution_tokens = self.tokenizer.encode(
            "# Solution", add_special_tokens=False
        )[1:]
        self.scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[
            1:
        ]
        self.eos_token = self.tokenizer.eos_token_id

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:
        single_beam = condense_newlines(single_beam)
        input_ids = self.tokenizer.encode(single_beam)

        begin_solution_flag = False

        candidate_positions = []

        temp_pos = []

        for start_idx in range(len(input_ids)):
            if tuple(
                input_ids[start_idx : start_idx + len(self.begin_solution_tokens)]
            ) == tuple(self.begin_solution_tokens):
                begin_solution_flag = True

            if begin_solution_flag and tuple(
                input_ids[start_idx : start_idx + len(self.scoring_tokens)]
            ) == tuple(self.scoring_tokens):
                candidate_positions.append(start_idx)
                temp_pos.append(self.tokenizer.batch_decode([input_ids[start_idx-2 : start_idx + len(self.scoring_tokens)]]))

            if input_ids[start_idx] == self.eos_token:
                candidate_positions.append(start_idx)
                temp_pos.append(self.tokenizer.batch_decode([input_ids[start_idx]]))
                break

        if not candidate_positions:
            return 0
        del candidate_positions[0]

        if not candidate_positions:
            return 0

        input_tensor = torch.tensor([input_ids]).to(self.device)
        candidate_positions_tensor = torch.tensor(candidate_positions)

        ###
        # print(temp_pos)
        ###

        with torch.no_grad():
            logits = self.model(input_tensor).logits

            scores = logits.mean(dim=-1)

            step_scores = scores[0][candidate_positions_tensor]

            step_probs = torch.sigmoid(step_scores).tolist()

        # print(repr(single_beam))
        # print(len(single_beam))
        # print(self.tokenizer.batch_decode(input_ids[102 : 102 + len(self.scoring_tokens)]))
        # print(len(step_probs))
        # print(step_probs)
        # print(candidate_positions)
        # print(self.tokenizer.batch_decode(input_ids[108 : 108 + len(self.scoring_tokens)]))
        # input()

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
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """

        result = []

        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result