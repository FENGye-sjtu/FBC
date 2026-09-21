"""Separate dataset task content from incompatible answer-only instructions."""
import re


def task_question(question):
    for directive in (
        'Please answer directly with only the letter of the correct option and nothing else.',
        'Please answer directly with a single word or number.',
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


def cot_prompt(question):
    return (task_question(question) + '\nReason step by step. Start with a short introduction, '
            'separate numbered reasoning steps by blank lines, and finish with FINAL ANSWER: '
            + answer_format(question))
