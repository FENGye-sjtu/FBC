"""Separate dataset task content from incompatible answer-only instructions."""
import re


def task_question(question):
    for directive in (
        'Please answer directly with only the letter of the correct option and nothing else.',
        'Please answer directly with a single word or number.',
        'Please select the correct answer from the options above.',
    ):
        question = question.replace(directive, '')
    return question.strip()


def answer_format(question):
    if re.search(r'(?m)^\s*A[.)]\s', question):
        return 'Return only the option letter, not the option text.'
    stem = task_question(question).lstrip()
    if re.match(r'(?i)^(is|are|can|does|do|has|have|was|were|will|would|could|should)\b', stem):
        return 'Return Yes or No, not 1 or 0.'
    return 'Return a short word, object name, or number that answers the question.'


def cot_prompt(question, style='reasoned_v2'):
    """Build declared CoT variants without changing the benchmark content.

    ``reasoned_v2`` is retained byte-for-byte for reproducibility.  The v3
    variants test the hypothesis that long, generic CoT causes needless visual
    speculation on RealWorldQA.  Both still expose blank-line-separated evidence
    units for FaithAct and the online verifier.
    """
    if style == 'reasoned_v2':
        return (task_question(question) + '\nReason step by step. Start with a short introduction, '
                'separate numbered reasoning steps by blank lines, and finish with FINAL ANSWER: '
                + answer_format(question))
    instruction = (
        'Inspect only the visual evidence needed to choose among the given options. '
        'Write 1 to 3 short numbered evidence statements, separating them with blank lines. '
        'Do not add an introduction, restate all options, invent hidden details, or make an '
        'unsupported numerical estimate. Stop reasoning once the visible evidence is sufficient. '
        'Then finish on a new line with FINAL ANSWER: ' + answer_format(question))
    if style == 'grounded_concise_v3':
        return task_question(question) + '\n' + instruction
    if style == 'preserve_select_v3':
        # Retain VLMEvalKit's ordinary MCQ instruction and add only a concise
        # evidence trace requirement. This measures whether the 70.46% direct
        # behavior can be preserved while producing inspectable steps.
        return question.strip() + '\nBefore selecting the answer, ' + instruction[0].lower() + instruction[1:]
    raise ValueError(f'Unknown CoT prompt style: {style}')
