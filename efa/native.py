"""Conservative conversion of explicit property questions into neutral atoms."""
import re


def property_atom(question,value):
    patterns=[r'^What is the (?:primary )?(color|shape|material) of (.+)\?$',
              r'^What (color|shape|material) is (.+)\?$']
    match=next((m for pattern in patterns if (m:=re.match(pattern,question))),None)
    if not match:
        return None
    predicate,subject=match.groups()
    # Do not turn a disputed color into a router identity.
    if predicate=='color' and re.search(r'\b(red|blue|green|yellow|pink|purple|black|white|brown|orange|beige)\b',subject,re.I):
        return None
    return dict(subject=subject,predicate=predicate,value=value,family='attribute',kind='visual')
