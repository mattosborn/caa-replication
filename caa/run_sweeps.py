import torch as t
import json
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset, behaviours, layers

class BatchedDataset:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            try:
                batch.append(next(self.dataset))
            except StopIteration:
                break
        if len(batch) == 0:
            raise StopIteration
        return batch
    
def run_sweeps(model_name: str):
    model = ModelWrapper(model_name)
    multipliers = [-1, 1]

    steering_results = { behaviour: {} for behaviour in behaviours }
    
    for behaviour in behaviours[:1]:
        dataset = load_dataset(f"{behaviour}_test_ab")
        
        for layer in layers[10: 20]:
            # print('layer', layer)
            layer_results = {m: [] for m in multipliers}
            steering_vectors = t.load(f'data/{behaviour}_steering_vectors.pt')

            for batch in tqdm(BatchedDataset(iter(dataset), 32)):
                prompts = [model.create_prompt_str(data['question'], '(') for data in batch]
                # prompt_lengths = model.tokenizer(prompts, return_length=True)['length']
                batch_prompt = model.tokenize_batch(prompts)
                
                for m in multipliers:
                    steered_output = model.prompt_with_steering(batch_prompt, steering_dict={layer: steering_vectors[layer] * m})
                    probabilities = model.calc_batch_probs(steered_output.logits, batch_prompt, batch)
                    
                    layer_results[m].extend(probabilities)
            
            # print('results', results)
            
            steering_results[behaviour][layer] = {m: sum(layer_results[m]) / len(layer_results[m]) for m in multipliers}

    # print(steering_results)
    with open(f'results/{model.get_model_name()}_layer_sweep.json', 'w') as f:
        json.dump(steering_results, f, indent=2)
