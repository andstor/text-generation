
from typing import List
from transformers import StoppingCriteriaList
from ..stopping_criterias import BraceMatchingCriteria
from ..arguments.generation_args import GenerationArguments

def create_stopping_criteria_list(args: GenerationArguments, **kwargs) -> StoppingCriteriaList:
    # Define stopping criterias
    stopping_criteria_list = StoppingCriteriaList()
    if args.use_brace_matching:
        stopping_criteria_list.append(BraceMatchingCriteria(tokenizer=kwargs["tokenizer"], start_level=args.brace_matching_start_level))
    
