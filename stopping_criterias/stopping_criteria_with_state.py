import torch
from transformers import StoppingCriteria

class StoppingCriteriaWithState(StoppingCriteria):
    def __init__(self):
        self.state = {"done": False}

    def should_stop(self):
        raise NotImplementedError

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteriaWithState needs to be subclassed")
