from box_stopping_criteria import BoxStoppingCriteria
from model_interface import BaseModel
from .prm_interface import PRM
from search_interface import Search, SearchOutput


class BeamSearch(Search):
    def __init__(
        self,
        model: BaseModel,
        prm: PRM,
        num_beams: int = 3,
        new_samples_per_beam: int = 3,
        max_expansion_rounds: int = 20,
        max_new_tokens: int = 100,
    ) -> None:
        """
        Initialize the BeamSearch class.

        Args:
            model (BaseModel): Model used for generating reasoning steps.
            prm (PRM): Process reward model used for scoring reasoning steps.
            num_beams (int): The number of diverse beams to sample.
            new_samples_per_beam (int): The number of next-step proposals to sample per beam.
            max_expansion_rounds (int): Maximum rounds of beam expansion before stopping.
            max_new_tokens (int): Max new tokens for each generation step.
        """
        self.model = model
        self.prm = prm
        self.num_beams = num_beams
        self.max_expansion_rounds = max_expansion_rounds

        # TODO Perhaps set `do_sample = True` later
        self.generation_config = dict(
            max_new_tokens=max_new_tokens,
            max_length=None,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            stopping_criteria=[BoxStoppingCriteria(
                self.model.tokenizer,
                self.model.get_delimeter_ids(),
            )],
        )

        self.inital_generation_config = dict(
            max_new_tokens=max_new_tokens,
            max_length=None,
            num_beams=new_samples_per_beam,
            num_return_sequences=new_samples_per_beam,
            stopping_criteria=[BoxStoppingCriteria(
                self.model.tokenizer,
                self.model.get_delimeter_ids(),
            )],
        )

    def __call__(self, problem: str) -> SearchOutput:
        """Solve the problem by generating reasoning steps using beam search."""
        beams = self.model.generate(
            problem,
            config=self.inital_generation_config,
            add_meta_prompt=True,
        )

        for i in range(self.max_expansion_rounds):
            new_beams = []

            for i in range(self.num_beams):
                if self.model.is_complete(beams[i]):
                    new_beams.append(beams[i])
                else:
                    new_sample_beams = self.model.generate(
                        beams[i], config=self.generation_config
                    )
                    new_beams.extend(new_sample_beams)

            step_scores = self.prm(new_beams)

            sorted_enumerated_step_scores = sorted(
                enumerate(step_scores),
                key=lambda x: x[1].score,
                reverse=True,
            )

            top_idxs = [i for i, _ in sorted_enumerated_step_scores[: self.num_beams]]
            beams = [new_beams[i] for i in top_idxs]

            if all(self.model.is_complete(beam) for beam in beams):
                break

        filtered_sorted_beams = [
            x[1].step
            for x in sorted_enumerated_step_scores
            if self.model.is_complete(x[1].step)
        ]

        if len(filtered_sorted_beams) == 0:
            return SearchOutput()

        return SearchOutput(
            answer=self.model.extract_answer(filtered_sorted_beams[0]),
            output=filtered_sorted_beams[0],
            alternate_outputs=filtered_sorted_beams,
        )
