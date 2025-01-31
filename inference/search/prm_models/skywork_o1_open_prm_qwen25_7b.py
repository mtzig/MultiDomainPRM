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

from .skywork_o1_prm_inference_utils.prm_model import PRM_MODEL
from .skywork_o1_prm_inference_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class Qwen7bPRM(PRM):
    def __init__(
        self,
        aggregation: str = "full",#the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_path='/mnt/data/huggingface/models/models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B/snapshots/d28ac22e10adb604a66ecfb1a00a1f527d5d8c84') -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.good_token = '+' #token in the vocabulary set that indicates the probability of good step 
        self.bad_token = '-' #token in the vocabulary set that indicates the probability of bad step 
        self.step_tag = 'ки'# step deliminator to locate the logits to get the prm score for each step

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        self.model = PRM_MODEL.from_pretrained(model_path).eval()

        if not quantization_config:
            self.model.to(self.device)

        self.aggregation = aggregation

    def __call_single(self, single_beam: str) -> float | list[float]:
        
        question, steps = single_beam

        processed_data = [prepare_input(question, steps, tokenizer=self.tokenizer, step_token=self.step_tag)]
        input_ids, steps, reward_flags = zip(*processed_data)
        input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, self.tokenizer.pad_token_id, device=self.device)

        _, _, rewards = self.model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
        step_probs = derive_step_rewards(rewards, reward_flags)

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

# if __name__ == "__main__":

#     prm = Qwen7bPRM(
#                 aggregation="full", 
#             )
    
#     json_file_path = "/home/ec2-user/strawberry/prmMixedDomain/prmMixedDomain.json"###mixedomain format json file
#     data = read_json_file(json_file_path)

#     #organizing input cot data format based on different prm data formats
#     for each_data in tqdm(data):
#         for cot in each_data["chain_of_thoughts"]:
#             steps = cot["steps"]
#             steps = [step.replace(prm.step_tag, "") for step in steps]
#             updated_steps = []
#             for index, step in enumerate(steps):
#                 indexed_step = f"{step} {prm.step_tag}"
#                 updated_steps.append(indexed_step)
#             steps = updated_steps
#             question = each_data["question"].replace(prm.step_tag, "")

#             input_all = [question, "".join(steps)]
#             # steps_all = f"{question} " + "".join(steps)

#             rewards = prm([input_all])
#             # rewards = prm([steps_all])
#             cot["prm_reward"] = rewards[0].score[0]

#             print(cot['prm_reward'])
#             print(len(cot["prm_reward"]))
#             print(len(steps))
#             input()