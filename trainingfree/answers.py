"""Reference-blind extraction of committed answers from free-form model output.

Formatting is permissive; semantic ambiguity is not resolved by guessing. Only
the generated text and publicly supplied choices enter this module. In
particular, an explicit label wins over inconsistent option wording.
"""
import json
import re
import unicodedata


PARSER_VERSION = 'committed_answer_v5'
_LABEL = r'[A-E]'
_NUMBER_WORDS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty'.split(), map(str, range(21))))
_BOOL_WORDS = {'yep': 'yes', 'yeah': 'yes', 'nope': 'no', 'nah': 'no'}


def answer_source(row):
    """Prefer a model's non-null raw field, including an explicit empty string."""
    if row.get('raw') is not None:
        return 'raw'
    if row.get('answer') is not None:
        return 'answer'
    return 'missing'


def raw_answer(row):
    """Shared runtime/replay input; stored answers cannot replace empty raw output."""
    source = answer_source(row)
    return '' if source == 'missing' else str(row[source])


def choices_from_question(question):
    """Read line-separated or inline A--E options, without a reference label."""
    question = str(question or '')
    pattern = re.compile(r'(?<!\w)(?:\(([A-E])\)|([A-E])[.):])\s+')
    found = list(pattern.finditer(question))
    # A contiguous ordered list avoids treating letters in question prose as
    # choices. The prompt suffix is excluded from the final option's wording.
    starts = [i for i, match in enumerate(found) if (match[1] or match[2]) == 'A']
    for start in reversed(starts):
        sequence = []
        for match in found[start:]:
            if (match[1] or match[2]) != chr(65 + len(sequence)):
                break
            sequence.append(match)
        if len(sequence) < 2:
            continue
        choices = {}
        for i, match in enumerate(sequence):
            end = sequence[i + 1].start() if i + 1 < len(sequence) else len(question)
            value = question[match.end():end].strip()
            value = re.split(r'\n\s*(?:Please\s+select|Return\s+only|Answer\s+with|'
                             r'FINAL\s+ANSWER|Choose\s+the)\b', value, flags=re.I)[0]
            choices[match[1] or match[2]] = value.strip()
        return choices
    return {}


def _clean(text):
    text = unicodedata.normalize('NFKC', str(text)).strip()
    text = re.sub(r'```(?:json|text|markdown)?\s*|```', '', text, flags=re.I)
    text = re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('**', '').replace('__', '').replace('`', '')
    text = text.strip(' \t\r\n*$')
    if len(text) >= 2 and (text[0], text[-1]) in {('"', '"'), ("'", "'"), ('“', '”')}:
        text = text[1:-1].strip()
    return text


def _normal(text):
    return re.sub(r'[^\w]+', ' ', _clean(text).casefold()).strip()


def _option_normal(text):
    # Lexical equivalents only; no model judge, embedding, or semantic guess.
    value = ' '.join(_NUMBER_WORDS.get(word, word) for word in _normal(text).split())
    return _BOOL_WORDS.get(value, value)


def _conditional_prefix(text, position):
    prefix = re.split(r'[\n.!?]|\bbut\b', text[:position], flags=re.I)[-1]
    return bool(re.search(r'\b(?:if|whether|suppose|assuming|hypothetically)\b', prefix, re.I))


def _events(text):
    """Find explicit answer commitments, keeping source position and strength."""
    events = []
    # Structured final-answer fields are stronger than incidental prose.
    for match in re.finditer(r'<(final_answer|final|answer)\b[^>]*>(.*?)</\1\s*>',
                             text, flags=re.I | re.S):
        events.append((match.start(), 3, match[2]))
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            for key in ('final_answer', 'final answer', 'answer', 'choice', 'option'):
                candidate = value.get(key)
                if isinstance(candidate, (str, int, float)):
                    events.append((match.start(), 3, str(candidate)))
                    break
    for match in re.finditer(r'\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}', text):
        events.append((match.start(), 3, match[1]))
    marker = re.compile(
        r'\b(?:(?P<final>final)\s+(?:answer|choice|option)|'
        r'(?:correct|best|most\s+(?:reasonable|plausible))\s+(?:answer|choice|estimate)|'
        r'answer|(?:I\s+)?(?:choose|select))'
        r'(?:\s+related\s+to\s+the\s+question)?\s*'
        r'(?:(?:is|would\s+be)\s*[:=]?\s*|[:=]\s*)', re.I)
    for match in marker.finditer(text):
        if _conditional_prefix(text, match.start()):
            continue
        before = text[max(0, match.start() - 40):match.start()]
        if re.search(r'(?:not|wrong|incorrect|reject(?:ed)?|example|format|other)\s*[:=]?\s*$',
                     before, re.I):
            continue
        strength = 3 if match['final'] else (1 if 'estimate' in match[0].lower() else 2)
        events.append((match.start(), strength, text[match.end():]))
    for match in re.finditer(r'\bI\s+(?:choose|select)\s+(?=(?:option\s+)?[A-E]\b)', text):
        events.append((match.start(), 2, text[match.end():]))
    for match in re.finditer(r'(?<!\w)(?:[Oo]ption\s+)?\(?([A-E])\)?\s+is\s+'
                             r'(?:the\s+)?(?:correct|final|best)\s*(?:answer|choice|option)?\b', text):
        if not _conditional_prefix(text, match.start()):
            events.append((match.start(), 2, match[1]))
    # Natural conclusions support option wording, but do not override an
    # explicit final answer merely because they occur in its explanation.
    for match in re.finditer(r'\b(?:Thus|Therefore|Hence|So)\s*,\s*|'
                             r'\bGiven\s+(?:these|this)\s+(?:observations|points|information)\s*,\s*',
                             text, re.I):
        events.append((match.start(), 1, text[match.end():]))
    return events


def _leading_label(text, choices, explicit=False):
    text = _clean(text)
    text = re.sub(r'^(?:the\s+)?(?:option|choice)\s*[:=]?\s*', '', text, flags=re.I)
    if explicit:
        text = re.sub(r'^(?:(?:I\s+(?:think|believe)\s+)|(?:probably|likely)\s+)', '', text, flags=re.I)
    match = re.match(r'^(?:\(([A-Ea-e])\)|\[([A-Ea-e])\]|([A-Ea-e]))(?=$|[\s.,:;)!?])', text)
    if not match:
        return None
    label = next(v for v in match.groups() if v).upper()
    if choices and label not in choices:
        return None
    remainder = text[match.end():]
    # Lowercase unbracketed "a car" is prose, not the label A.
    if match[3] == 'a' and remainder.strip(' .,:;!?'):
        return None
    if re.match(r'^\s*(?:[/&]|,?\s*(?:or|and)\b)', remainder, re.I):
        return None
    if re.match(r'^\s*(?:or|and|is\s+(?:not|wrong|incorrect)|'
                r'(?:would|could|might)\s+be|isn.t)\b', remainder, re.I):
        return None
    if re.match(r'^\s*[,/]?\s*[A-E]\s*(?:or|/|and)\b', remainder):
        return None
    # A letter followed by prose needs label punctuation, a line break, or an
    # explicit commitment. This prevents the article in "A red car ...".
    if remainder and not re.match(r'^[.):,;!?\n\r]|^\s*[-–—]|^\s*$', remainder):
        if not explicit or not re.match(r'^\s+(?:because|since|is\s+correct)\b', remainder, re.I):
            return None
    first_line = text.splitlines()[0]
    if len(re.findall(r'(?<!\w)[A-E][.):]\s', first_line)) > 1:
        return None
    return label


def _option_wording(text, choices, explicit=False):
    """Map positively asserted option wording, never a mere/negated mention."""
    text = _clean(text)
    normalized = _option_normal(text)
    exact = [key for key, value in choices.items() if _option_normal(value) == normalized]
    if len(exact) == 1:
        return exact[0]
    # A conclusion is a single answer clause; later reasoning does not turn
    # its examples or rejected alternatives into the selected answer.
    clause = re.split(r'\n|[!?]|(?<!\d)\.(?!\d)', text, maxsplit=1)[0]
    if re.search(r'\b(?:options?\s+(?:are|include)|consider(?:ed|ing)?|mention(?:ed)?|'
                 r'cannot|can.t|unable|impossible|not\s+possible|neither|'
                 r'none\s+of|unclear|undetermined|either|possible|possibility|'
                 r'hypothetical|candidate|example|\bif\b)\b', clause, re.I):
        return None
    if not explicit and (len(text.splitlines()) > 1 or len(text) > 300):
        return None
    clause = _option_normal(clause)
    candidates = []
    for label, value in choices.items():
        value = _option_normal(value)
        if not value:
            continue
        for match in re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)', clause):
            before, after = clause[:match.start()], clause[match.end():]
            # Limit to the current contrast clause: "not red but green" can
            # affirm green, whereas "red is not visible" cannot affirm red.
            before = re.split(r'\b(?:but|however|rather)\b', before)[-1]
            if re.search(r'\b(?:no|not|without|except|isn t|isnt)\b(?:\s+\w+){0,4}\s*$', before):
                continue
            if re.match(r'\s+(?:(?:is|are|was|were)\s+)?(?:not|absent|wrong|incorrect|unlikely)\b', after):
                continue
            candidates.append((label, match.start(), match.end()))
    # Prefer a fully specified choice to its substring (e.g. bottom left over
    # left). Two separate positive choices remain ambiguous.
    candidates = [item for item in candidates if not any(
        other[1] <= item[1] and other[2] >= item[2] and other[2] - other[1] > item[2] - item[1]
        for other in candidates)]
    labels = {item[0] for item in candidates}
    return next(iter(labels)) if len(labels) == 1 else None


def _embedded_label(text, choices):
    """Accept one asserted inline choice, before any option-wording inference.

    This is a formatting rule for e.g. "the count is B. 2", independent of
    which option is correct. "the count is A. 2" must therefore remain A even
    if option B has wording "2". Negated, conditional and multiple choices do
    not provide an unambiguous commitment.
    """
    clause = _clean(text).split('\n', 1)[0]
    matches = list(re.finditer(
        r'\b(?:is|are|equals?|would\s+be)\s+(?:option\s+)?'
        r'(?P<answer>\([A-E]\)|\[[A-E]\]|[A-E])'
        r'(?=$|[.):,;!?]|\s*[-–—])', clause))
    if not matches:
        return None
    # Empty means an inline label was present but uncommitted/ambiguous. The
    # caller must not then recover an option from its accompanying wording.
    if len(matches) != 1 or re.search(
            r'\b(?:if|whether|suppose|assuming|hypothetically|not|no|never|'
            r'neither|wrong|incorrect|false|rejected|example|format|possible|'
            r'candidate|considered)\b|\b(?:isn.t|aren.t|can.t)\b', clause, re.I):
        return ''
    match = matches[0]
    label = match['answer'].strip('()[]')
    declared = re.findall(r'(?<!\w)(?:\(([A-E])\)|\[([A-E])\]|([A-E])[.):])', clause)
    mentioned = {letter for item in declared for letter in item if letter}
    if mentioned - {label} or label not in choices:
        return ''
    tail = clause[match.end():]
    if re.search(r'\b(?:or|and)\s+(?:option\s+)?[A-E]\b|[/&]\s*[A-E]\b', tail):
        return ''
    return label


def _terminal_option(text, choices):
    """Recognize a standalone label plus its complete public option wording.

    Used only when no stronger answer field exists. The preceding task may be
    conditional ("if you stay in this lane ..."); a rejected/hypothetical option
    or an option line carrying its own condition is not a final commitment.
    None means this formatting rule does not apply; empty means reject the
    apparent candidate without recovering its wording through another rule.
    """
    lines = [_clean(line) for line in text.splitlines() if _clean(line)]
    if len(lines) < 2:
        return None
    pattern = r'^(?:\(([A-E])\)[.):]?|\[([A-E])\][.):]?|([A-E])[.):])\s*(.+?)\s*$'
    match = re.fullmatch(pattern, lines[-1])
    if not match:
        return None
    label = next(item for item in match.groups()[:3] if item)
    wording = match[4]
    listed = [re.fullmatch(pattern, line) for line in lines]
    labels = {next(item for item in candidate.groups()[:3] if item)
              for candidate in listed if candidate}
    if len(labels) != 1:
        return ''
    context = lines[-2]
    if re.search(r'\b(?:rejected|excluded|incorrect|wrong|alternatives?|candidates?|'
                 r'possibilities|example|hypothetical)\b|'
                 r'\b(?:might|could|would)\s+(?:instead\s+)?be\b|'
                 r'\b(?:not|never)\s+(?:the\s+)?(?:answer|chosen|selected|correct)\b',
                 context, re.I):
        return ''
    # Exact option text can itself contain a negation, e.g. "Not enough
    # information". Reject added qualifiers, not legitimate choice wording.
    if label in choices and _option_normal(wording) == _option_normal(choices[label]):
        return label
    if re.search(r'\b(?:if|unless|otherwise|not|incorrect|wrong|rejected|might|could|'
                 r'maybe|perhaps)\b|\b(?:or|and)\s+(?:option\s+)?[A-E]\b', wording, re.I):
        return ''
    # Inconsistent label/wording is not silently remapped to a different label.
    return None


def _conclusion_option_line(text, choices):
    """Recover a labeled answer line followed by explanation, not an option list.

    Some CoT outputs commit to ``C. ...`` after an answer-introducing sentence
    and then continue with prose.  A last-line-only parser misses these.  The
    preceding sentence must denote a singular selection, and the answer line
    must be the sole unbulleted choice line after it.
    """
    lines = [_clean(line) for line in str(text).splitlines() if _clean(line)]
    marker = re.compile(
        r'\b(?:answer|choice|option|action|response|selection)\b.*'
        r'\b(?:is|would\s+be)\s*:\s*$', re.I)
    label_line = re.compile(r'^([A-E])[.)]\s+\S')
    for index in range(len(lines) - 2, -1, -1):
        if not marker.search(lines[index]):
            continue
        if re.search(r'\b(?:rejected|wrong|incorrect|not|if|hypothetical|'
                     r'alternative|possible|candidate|might|could)\b', lines[index], re.I):
            continue
        match = label_line.match(lines[index + 1])
        if not match or match[1] not in choices:
            continue
        if re.search(r'\b(?:if|unless|otherwise|incorrect|wrong|rejected|might|could)\b|'
                     r'\b(?:or|and)\s+(?:option\s+)?[A-E]\b', lines[index + 1], re.I):
            continue
        if any(label_line.match(line) for line in lines[index + 2:]):
            continue
        return match[1]
    return None


def _extract_v2(text, question=''):
    """Preserved v2 extraction path, including its existing limitations.

    Explicit final fields take precedence over a stray terminal letter. An
    unresolved explicit final field blocks fallback to an earlier guess.
    Neither this signature nor any helper accepts the reference answer.
    """
    if text is None:
        return ''
    original = str(text).strip()
    if not original:
        return ''
    choices = choices_from_question(question)
    events = _events(original)
    explicit = bool(events)
    strength = 0
    if events:
        _, strength, selected = max(events, key=lambda event: (event[1], event[0]))
        explicit = strength >= 2
    else:
        selected = original
    selected = _clean(selected)
    selected = re.sub(r'^(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is)?\s*:?\s*',
                      '', selected, flags=re.I)
    selected = _clean(selected)
    if not selected:
        return ''
    if choices and strength < 2:
        lines = [line.strip() for line in selected.splitlines() if line.strip()]
        bare = [re.fullmatch(r'[\[(]?[A-Ea-e][\])]?[.!]?', line) for line in lines]
        if all(bare) and len({_clean(line).strip('()[].!').upper() for line in lines}) > 1:
            return selected
        # With ordinary prose conclusions, a dedicated final label is the
        # model's most direct answer, including when its reasoning disagrees.
        if bare[-1]:
            return _leading_label(lines[-1], choices) or selected
    label = _leading_label(selected, choices, explicit=explicit)
    if label and (choices or re.fullmatch(r'[\[(]?[A-Ea-e][\])]?.?', selected)):
        return label
    if choices:
        embedded = _embedded_label(selected, choices)
        if embedded is not None:
            return embedded or selected.strip(' \t\r\n*`.')
        wording = _option_wording(selected, choices, explicit=bool(events))
        if wording:
            return wording
        if not events:
            # A standalone final line is a conventional answer field. Do not
            # apply this fallback when a stronger explicit answer was given.
            final = _clean(selected.splitlines()[-1])
            if re.fullmatch(r'[\[(]?[A-Ea-e][\])]?[.!]?', final):
                return _leading_label(final, choices) or selected
            if len(selected.splitlines()) > 1:
                final_label = _leading_label(final, choices)
                if final_label:
                    return final_label
    return selected.strip(' \t\r\n*`.')


def extract_answer(text, question=''):
    """Add reference-blind formatting recoveries without semantic guessing.

    Already parsed public choice labels are immutable under this extension.
    Only unresolved v2 outputs with no strong final-answer field are eligible.
    This preserves v2's known false-positive limitations as well as its valid
    answers; the extension does not reinterpret those results with references.
    """
    previous = _extract_v2(text, question)
    choices = choices_from_question(question)
    if not choices or text is None:
        return previous
    original = str(text).strip()
    # Some generations bold only the marker (``**Final Answer**: B``).  The
    # closing Markdown delimiter used to sit between ``answer`` and ``:``, so
    # a marker at the end of a long rationale was invisible to v2/v3.  Remove
    # only that delimiter and replay the same parser; no option or reference
    # content is consulted by this formatting-only recovery.
    normalized_marker = re.sub(
        r'(?i)(\bfinal\s+(?:answer|choice|option))\s*(?:\*\*|__)+\s*([:=])',
        r'\1\2', original)
    if normalized_marker != original:
        recovered = _extract_v2(normalized_marker, question)
        if recovered in choices:
            return recovered
    if previous in choices:
        return previous
    events = _events(original)
    if events:
        _, strength, selected = max(events, key=lambda event: (event[1], event[0]))
        if strength >= 2:
            return previous
    else:
        selected = original
    conclusion = _conclusion_option_line(original, choices)
    if conclusion:
        return conclusion
    terminal = _terminal_option(_clean(selected), choices)
    return terminal if terminal in choices else previous
