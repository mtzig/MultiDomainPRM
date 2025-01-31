import math,re
import statistics
from typing import Optional
from peft import PeftModel
import torch
from .prm_interface import PRM, StepScore
from torch.types import Device
from transformers import (  # type: ignore  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import sys
sys.path.append("api_model_single_call")
from gpt_agent import gpt_agent

class gpt4o_prm(PRM):
    def __init__(
        self,
        aggregation: str = "full",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        debug = True,
    ) -> None:

        self.aggregation = aggregation
        self.debug = debug

        self.system_prompt = """
        You are an experienced evaluator specializing in assessing the quality of reasoning steps in problem-solving. Your task is to analyze and classify the last step in a student's solution based on its correctness, clarity, and contribution to the reasoning process.

        #### Task Description
        You will be provided with:
        1. A Question
        2. A Student's Step-by-Step Solution, where each step begins with `##Solution##` and ends with a newline character.

        The provided solution may be partial or incomplete, so evaluate correctness only up to the given steps. Do not assume any unstated steps or context unless explicitly provided.

        Your job is to analyze the last step of the solution and classify it as one of the following:
        - GOOD
        - OK
        - BAD

        #### Criteria for Classification
        1. GOOD Step
        A step is classified as GOOD if it meets all of these criteria:
        - Correct: Everything stated is accurate and aligns with known principles or the given problem.
        - Verifiable: The step can be verified using common knowledge, simple calculations, or a quick reference (e.g., recalling a basic theorem). If verifying requires extensive effort (e.g., detailed calculations or obscure references), mark it BAD instead.
        - Appropriate: The step fits logically within the context of the preceding steps. If a prior mistake exists, a GOOD step can correct it.
        - Insightful: The step demonstrates reasonable problem-solving direction. Even if ultimately progress in the wrong direction, it is acceptable as long as it represents a logical approach.

        2. OK Step
        A step is classified as OK if it is:
        - Correct and Verifiable: Contains no errors and can be verified.
        - Unnecessary or Redundant: Adds little value, such as restating prior information or providing basic encouragement (e.g., “Good job!”).
        - Partially Progressing: Makes some progress toward the solution but lacks decisive or significant advancement.

        3. BAD Step
        A step is classified as BAD if it:
        - Is Incorrect: Contains factual errors, misapplies concepts, or derives an incorrect result.
        - Is Hard to Verify: Requires significant effort to confirm due to poor explanation.
        - Is Off-Topic: Includes irrelevant or nonsensical information.
        - Derails: Leads to dead ends, circular reasoning, or unreasonable approaches.

        #### Output Guidelines
        - Analyze the solution up to the last step, considering only the provided information.
        - Provide a step-by-step reasoning analysis of the last step's quality.
        - Conclude your analysis with the classification: STEP_CLASS: GOOD, STEP_CLASS: OK, or STEP_CLASS: BAD.

        #### Output Format Example
        [Step-by-step reasoning about how to classify the last step]
        STEP_CLASS: [GOOD | OK | BAD]
        """

        
        self.gpt4o_evaluator = gpt_agent(role_description=self.system_prompt, model_name= "gpt-4o", temperature=1.0)



        

    def __call_single(self, single_beam: str) -> float | list[float]:
        
        evaluation = self.gpt4o_evaluator(single_beam)

        try:
            # error_idx = int(error_idx)
            pattern = r"STEP_CLASS:\s*(GOOD|OK|BAD)"
            match = re.search(pattern, evaluation)
            step_class =  -1 if match.group(1) == 'BAD' else 1
            
        except:
            print('output not in the correct format. Default labeling it -1')
            step_class = -3

        return float(step_class), evaluation
    

    def __call__(self, steps: list[str]):
        """
        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of dictionaries where each dictionary
        """
        result = []

        for beam in steps :
            step_score, evaluation_step = self.__call_single(beam)
            if self.debug:
                result.append((StepScore(step=beam, score=step_score), evaluation_step))
            else:
                result.append(StepScore(step=beam, score=step_score))

        return result

if __name__ == "__main__":
    pass



