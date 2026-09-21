"""Optional common final-answer format for controlled comparisons, not released prompts."""
from .prompts import answer_format


class FinalFormatAdapter:
    def __init__(self, vlm, question):
        self.vlm, self.question = vlm, question

    def generate(self, image, prompt, **kwargs):
        # Leave FaithAct's text-only noun extraction exactly unchanged.
        if image is not None:
            prompt += '\nFinish with FINAL ANSWER: ' + answer_format(self.question)
        return self.vlm.generate(image,prompt,**kwargs)
