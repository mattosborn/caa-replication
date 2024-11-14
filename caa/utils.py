import json
from typing import Dict, TypedDict, TypeVar, List, Generic, Any
import torch as t
import os


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

TestConfig = TypedDict('TestConfig', {'behaviour': str, 'multiplier': float, 'layer': int, 'model': str})

Results = List[Dict[str, Any]]

def get_model_filename(model_name: str) -> str:
    model_shorthand_dict = {
        'meta-llama/Llama-2-7b-chat-hf': '7b-chat-hf',
    }
    return model_shorthand_dict.get(model_name, model_name)

def get_result_filename(config: TestConfig) -> str:
    model_name = get_model_filename(config['model'])
    return f'results/raw/{model_name}_{config["behaviour"]}_{config["layer"]}_{config["multiplier"]}.json'

def write_results(config: TestConfig, results: Results):
    filename = get_result_filename(config)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(get_result_filename(config), 'w') as f:
        json.dump(results, f, indent=2)

def read_results(config: TestConfig) -> Results:
    with open(get_result_filename(config), 'r') as f:
        return json.load(f)