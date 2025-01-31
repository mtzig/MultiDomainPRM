from box_stopping_criteria import BoxStoppingCriteria
from model_interface import BaseModel
from .prm_interface import PRM
from search_interface import Search, SearchOutput


class BestOfN(Search):
    def __init__(
        self,
        model: BaseModel,
        prm: PRM,
        width: int = 5,
        max_new_tokens: int = 2000,
    ) -> None:
        """
        Initialize the BestOfN class.

        Args:
            model (BaseModel): Model used for generating reasoning steps.
            prm (PRM): Process reward model used for scoring reasoning steps.
            width (int): The number trajectories (n).
            max_new_tokens (int): Max new tokens for each trajectory.
        """
        self.model = model
        self.prm = prm

        self.generation_config = dict(
            max_new_tokens=max_new_tokens,
            max_length=None,
            num_beams=width,
            num_return_sequences=width,
            stopping_criteria=[BoxStoppingCriteria(
                self.model.tokenizer, 
                [self.model.end_solution_id],
            )],
        )

    def __call__(self, problem: str) -> SearchOutput:
        beams = self.model.generate(
            problem,
            config=self.generation_config,
            add_meta_prompt=True,
        )

        filtered_beams = [b for b in beams if self.model.is_complete(b)]
        if len(filtered_beams) == 0:
            return SearchOutput()

        scored_steps = self.prm(filtered_beams)
        final_solution = max(scored_steps, key=lambda x: x.score)
        return SearchOutput(
            answer=self.model.extract_answer(final_solution.step),
            output=final_solution.step,
            alternate_outputs=[b for b in filtered_beams],
        )
