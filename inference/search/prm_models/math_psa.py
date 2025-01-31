import math
import statistics
from typing import Optional
from tqdm import tqdm
import torch
from peft import PeftModel,PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
from .prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import json

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


class math_psa_prm(PRM):
    def __init__(
        self,
        aggregation: str = "full",#the way how prm step scores will be aggregated in a solution
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
    ) -> None:
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n' #ки
        self.step_tag2 = '\n\n'

        self.model_path = "Qwen/Qwen2.5-Math-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,add_eos_token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16).to(self.device)
        
        self.adapter_path = "/mnt/data/huggingface/models/Math-psa/checkpoint-2127"

        # self.adapter_config = PeftConfig.from_pretrained(self.adapter_path)

        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        if not quantization_config:
            self.model.to(self.device)
        self.aggregation = aggregation


        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left"  # Allow batched inference

        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [488, 481]
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] # 76325



    def __call_single(self, single_beam: str) -> float | list[float]:
        input_for_prm = single_beam
        # input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        # with torch.no_grad():
        #     logits = self.model(input_id).logits[:,:,self.candidate_tokens]
        #     scores = logits.softmax(dim=-1)[:,:,0]#for the good token
        #     step_scores = scores[input_id == self.step_tag_id]
       
        # step_probs  = step_scores.tolist()


        ###
        input_id = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            # print(logits)
            scores = logits.softmax(dim=-1)[:,:,0] 
            # print(scores)
            step_scores = scores[input_id == self.step_tag_id]
            step_probs  = step_scores.tolist()
        ###

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
    



if __name__ == "__main__":

    prm = math_psa_prm(
                aggregation="full", 
            )
    
    json_file_path = "/home/ec2-user/strawberry/prmMixedDomain/prmMixedDomain.json"###mixedomain format json file
    data = read_json_file(json_file_path)

    #organizing input cot data format based on different prm data formats
    for each_data in tqdm(data):
        for cot in each_data["chain_of_thoughts"]:
            steps = cot["steps"]
            steps = [step.replace("\n", "") for step in steps]
            question = each_data["question"].replace("\n", "")

            updated_steps = []
            for index, step in enumerate(steps):
                indexed_step = f"Step {str(index+1)}: {step} \n\n\n\n\n "
                updated_steps.append(indexed_step)
            steps = updated_steps
            steps_all = f"{question} " + "".join(steps)
            rewards = prm([steps_all])
            cot["prm_reward"] = rewards[0].score
            print(len(cot["prm_reward"]))

            print(len(steps))
            input()
    

