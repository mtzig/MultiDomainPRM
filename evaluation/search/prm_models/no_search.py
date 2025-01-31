from box_stopping_criteria import BoxStoppingCriteria
from model_interface import BaseModel
from search_interface import Search, SearchOutput


class NoSearch(Search):
    def __init__(self, model: BaseModel, max_new_tokens: int = 2000) -> None:
        """
        Initialize the NoSearch class.

        Args:
            model (BaseModel): Model used for generating reasoning steps.
            max_new_tokens (int): Max new tokens for each trajectory.
        """
        self.model = model

        self.generation_config = dict(
            max_new_tokens=max_new_tokens,
            max_length=None,
            num_beams=1,
            num_return_sequences=1,
            stopping_criteria=[BoxStoppingCriteria(
                self.model.tokenizer, 
                [self.model.end_solution_id],
            )],
        )

    def __call__(self, problem: str) -> SearchOutput:
        output = self.model.generate(
            problem,
            config=self.generation_config,
            add_meta_prompt=True,
        )[0]

        return SearchOutput(
            answer=self.model.extract_answer(output),
            output=output,
        )
