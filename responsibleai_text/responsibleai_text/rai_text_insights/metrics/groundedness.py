"""Groundedness metric."""

import datasets
import evaluate
import pandas as pd

logger = evaluate.logging.get_logger(__name__)


_CITATION = """
"""

_DESCRIPTION = """The groundedness metric.
"""

_KWARGS_DESCRIPTION = """
**SOME DESCRIPTION**
"""

_SYS_PROMPT = """
You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric.
Your response will be used in automated evaluation of question-answering systems, and must be an integer between 1 and 5, and nothing else.
""".strip()

_TEMPLATE = """
1. 5: The ANSWER follows logically from the information contained in the CONTEXT.
2. 1: The ANSWER is logically false from the information contained in the CONTEXT.
3. an integer score between 1 and 5 and if such integer score does not exists, use 1: It is not possible to determine whether the ANSWER is true or false without further information.
Read the passage of information thoroughly and select the correct answer from the three answer labels. Read the CONTEXT thoroughly to ensure you know what the CONTEXT entails.
Note the ANSWER is generated by a computer system, it can contain certain symbols, which should not be a negative factor in the evaluation.

CONTEXT:
{context}

ANSWER:
{prediction}
""".strip()


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Groundedness(evaluate.Metric):
    def _info(self):

        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence")
                }
            ),
        )

    def _compute(self, *, predictions=None, references=None, **kwargs):
        m = []
        templated_ques = []

        for p, r in zip(predictions, references):
            templated_ques.append(_TEMPLATE.format(context=r, prediction=p))

        model = kwargs['wrapper_model']

        inp = pd.DataFrame({
            'questions' : templated_ques,
            'sys_prompt' : _SYS_PROMPT})

        responses = model.predict(inp)

        for r in responses:
            try:
                m.append(int(r))
            except ValueError as e:
                logger.warning('Failed to parse metric `%s`: %s', r, e)
                m.append(0)
        return {'scores' : m}
            