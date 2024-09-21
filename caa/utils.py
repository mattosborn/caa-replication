import json
from typing import Dict, TypedDict, TypeVar, List, Generic
import torch as t


# behaviours = ['coordinate', 'corrigible', 'hallucination',
#               'myopic', 'refusal', 'survival', 'sycophancy']

behaviours = ["coordinate", "corrigible", "myopic", "refusal", "survival"]

layers = [i for i in range(32)]

T = TypeVar('T')

class BatchedDataset(Generic[T]):
    def __init__(self, dataset: List[T], batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_iter = iter(dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __next__(self):
        batch: List[T] = []
        for i in range(self.batch_size):
            try:
                batch.append(next(self.dataset_iter))
            except StopIteration:
                break
        if len(batch) == 0:
            raise StopIteration
        return batch
    

SteeringDict = Dict[int, t.Tensor]

class QuestionData(TypedDict):   
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str

def load_dataset(name) -> List[QuestionData]:
    dataset_file_name = f'datasets/{name}.json'
    dataset_file = open(dataset_file_name, 'r')
    dataset = json.load(dataset_file)
    return dataset
