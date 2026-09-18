"""Constrain serialization only; semantic values remain freely generated."""
ATOM_SCHEMA={'type':'object','properties':{
    'subject':{'type':'string','minLength':1},'predicate':{'type':'string','minLength':1},
    'value':{'type':'string','minLength':1},
    'family':{'type':'string','enum':['attribute','identity','text','relation','count','numeric','deduction']},
    'kind':{'type':'string','enum':['visual','premise']}},
    'required':['subject','predicate','value','family','kind'],'additionalProperties':False}
STEP_SCHEMA={'type':'object','properties':{
    'atoms':{'type':'array','items':ATOM_SCHEMA,'minItems':1,'maxItems':3},
    'final':{'type':'boolean'}},'required':['atoms','final'],'additionalProperties':False}
REVISIT_SCHEMA={'type':'object','properties':{'review_parent':{'type':'string','minLength':1}},
                'required':['review_parent'],'additionalProperties':False}
BOX={'type':'array','items':{'type':'integer','minimum':0},'minItems':4,'maxItems':4}
ROUTE_SCHEMA={'type':'object','properties':{
    'boxes':{'type':'array','items':BOX,'minItems':1,'maxItems':3},
    'context_boxes':{'type':'array','items':BOX,'minItems':1,'maxItems':3}},
    'required':['boxes','context_boxes'],'additionalProperties':False}


def step_parser():
    from lmformatenforcer import JsonSchemaParser,SequenceParser,StringParser,UnionParser
    from efa.format import STEP_END
    return SequenceParser([UnionParser([JsonSchemaParser(STEP_SCHEMA),JsonSchemaParser(REVISIT_SCHEMA)]),StringParser(STEP_END)])
