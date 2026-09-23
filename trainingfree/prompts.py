"""Separate dataset task content from incompatible answer-only instructions."""
import re

from .answers import choices_from_question


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


def compact_initial_prompt(question):
    """One tentative visual step; the controller still audits it before use."""
    choices = choices_from_question(question)
    stem = re.sub(r'^\s*Question\s*:\s*', '', task_question(question), flags=re.I)
    if choices:
        answer_domain = 'one of ' + ', '.join(choices)
    elif re.match(r'(?i)^(is|are|can|does|do|has|have|was|were|will|would|could|should)\b', stem):
        answer_domain = 'Yes or No'
    else:
        answer_domain = 'a short word or number'
    return (
        'Inspect the image and propose exactly one visual evidence step for this question.\n'
        'Return one JSON object using the supplied schema:\n'
        '- kind: "visual".\n'
        '- claim: one short sentence stating a relevant visible fact, not an option label '
        'or a full reasoning chain.\n'
        '- evidence_question: a short neutral question about what should be inspected, '
        'without suggesting its answer.\n'
        '- final_answer: if this observation is sufficient to answer the question, use '
        + answer_domain + '; otherwise use JSON null, not the string "null".\n'
        'The observation and any proposed answer are tentative and must pass the audit.\n\n'
        'Question: ' + task_question(question))


def cot_prompt(question, style='reasoned_v2'):
    """Build declared CoT variants without changing the benchmark content.

    ``reasoned_v2`` is retained byte-for-byte for reproducibility.  The v3
    variants test the hypothesis that long, generic CoT causes needless visual
    speculation on RealWorldQA.  Both still expose blank-line-separated evidence
    units for FaithAct and the online verifier.  The v4 variants scope answer
    formatting to the final line and reserve a separate header paragraph, since
    the released FaithAct reader discards the first and last paragraphs.
    """
    if style == 'reasoned_v2':
        return (task_question(question) + '\nReason step by step. Start with a short introduction, '
                'separate numbered reasoning steps by blank lines, and finish with FINAL ANSWER: '
                + answer_format(question))
    if style == 'faithact_gcot_official':
        # Released verbatim in FaithAct's template.py as TEMPLATE_GCOT.  Keep
        # this separate from our layout prompts: it is the missing upstream
        # initial-reasoning protocol, not a prompt tuned on our benchmark.
        payload = re.sub(r'^\s*Question\s*:\s*', '', task_question(question), flags=re.I)
        return (
            '\nYou are a visual reasoning assistant.\n'
            'Follow this process strictly:\n\n'
            'Step 1. Grounding: Identify and describe the region(s) in the image relevant to the question.\n'
            'Step 2. Reasoning: Use the grounded evidence to reason step by step.\n'
            'Step 3. Answer: Provide your final concise answer.\n\n'
            'Question: ' + payload + '\n'
        )
    if style == 'faithact_paper_cot_literal':
        # Sec. 3.1 states that the raw chain is produced by prompting "think
        # step by step". This is a separate paper-literal candidate; the
        # release does not reveal the missing Qwen adapter that applied it.
        return task_question(question) + '\nThink step by step.'
    if style == 'evidence_layout_v5':
        choices = choices_from_question(question)
        stem = re.sub(r'^\s*Question\s*:\s*', '', task_question(question), flags=re.I)
        if choices:
            final_value, placeholder = 'a single option letter', '<option>'
        elif re.match(r'(?i)^(is|are|can|does|do|has|have|was|were|will|would|could|should)\b', stem):
            final_value, placeholder = 'Yes or No', '<answer>'
        else:
            final_value, placeholder = 'a short word or number', '<answer>'
        return (task_question(question) + '\n\n'
                'Give one or two short numbered facts from the image that are needed to answer. '
                'Copy the blank-line layout below exactly, replacing the placeholders with your text. '
                'Evidence: must be its own paragraph. Leave an empty line after it, between '
                'numbered facts, and before FINAL ANSWER. Add a second fact only if needed, '
                'as a separate numbered paragraph. Do not output placeholders or a code fence. '
                'Only the final answer line should contain ' + final_value + '.\n\n'
                'Evidence:\n\n1. <visible fact>\n\nFINAL ANSWER: ' + placeholder)
    if style == 'discriminative_evidence_v6':
        choices = choices_from_question(question)
        final_value = 'one option letter' if choices else 'a short answer'
        return (
            task_question(question) + '\n\n'
            'Find the visible cue that actually distinguishes the offered answers. '
            'Inspect that cue in the full image before selecting an option; a generic '
            'related object is not enough. For counts, scan all distinct relevant '
            'instances; for left/right, check the referenced object and image-relative '
            'direction; for traffic actions, use an explicit visible sign or signal; '
            'for size, compare the same physical dimension. Do not infer hidden details '
            'or metric distance from image coordinates. State one or two short directly '
            'visible facts, then select the answer. Keep the blank-line layout exactly. '
            'Only the last line should contain ' + final_value + '.\n\n'
            'Evidence:\n\n1. <decisive visible fact>\n\nFINAL ANSWER: <answer>'
        )
    if style == 'discriminative_layout_v7':
        choices = choices_from_question(question)
        final_value, placeholder = ('a single option letter', '<option>') if choices else (
            'a short answer', '<answer>')
        return (
            task_question(question) + '\n\n'
            'Give one or two short numbered facts from the image that actually distinguish '
            'the offered answers. Inspect the relevant objects in the full image. For counts, '
            'scan distinct instances; for left/right, check the referenced object and direction; '
            'for traffic actions, require a visible sign or signal; for size, compare the same '
            'physical dimension. Do not infer hidden details or metric distance from image '
            'coordinates. Copy the blank-line layout below exactly, replacing the placeholders. '
            'Evidence: must be its own paragraph. Leave an empty line after it, between '
            'numbered facts, and before FINAL ANSWER. Add a second fact only if needed, '
            'as a separate numbered paragraph. Do not output placeholders or a code fence. '
            'Only the final answer line should contain ' + final_value + '.\n\n'
            'Evidence:\n\n1. <decisive visible fact>\n\nFINAL ANSWER: ' + placeholder
        )
    if style in ('evidence_brief_v4', 'evidence_adaptive_v4'):
        if re.search(r'(?m)^\s*A[.)]\s', question):
            final_value = 'the single option letter'
        elif re.match(r'(?i)^(is|are|can|does|do|has|have|was|were|will|would|could|should)\b',
                      task_question(question).lstrip()):
            final_value = 'Yes or No'
        else:
            final_value = 'a short word, object name, or number'
        evidence_instruction = (
            'Give one or two short numbered statements of the visible evidence that determines the answer.'
            if style == 'evidence_brief_v4' else
            'Give the minimum numbered evidence steps needed to answer. '
            'Use one step when the answer is clear; add steps only to resolve a relevant ambiguity. '
            'Base each step on what is visible in the image.')
        return (task_question(question) + '\n\n'
                + evidence_instruction + '\n'
                'Start with the literal header Evidence: as its own paragraph. '
                'Separate the header, each numbered evidence step, and the final answer with a blank line. '
                'Finish with FINAL ANSWER: <answer>, where <answer> is ' + final_value + '. '
                'This short-answer format applies only to the final line; include the evidence before it.')
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
