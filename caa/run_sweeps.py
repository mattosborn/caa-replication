import torch as t
import json
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset, behaviours, layers, BatchedDataset

def run_sweeps(model_name: str):
    model = ModelWrapper(model_name)
    multipliers = [-1, 1]

    steering_results = { behaviour: {} for behaviour in behaviours }
    
    for behaviour in behaviours[:2]:
        dataset = load_dataset(f"{behaviour}_test_ab")
        
        print(f"Running sweeps for {behaviour}")
        
        for layer in tqdm(layers[6: 20]):
            layer_results = {m: [] for m in multipliers}
            steering_vectors = t.load(f'data/{behaviour}_steering_vectors.pt')

            for batch in BatchedDataset(dataset, 32):
                for m in multipliers:
                    steered_output = model.batch_prompt(batch, steering_dict={layer: steering_vectors[layer] * m})
                    layer_results[m].extend(steered_output['probabilities'])
            
            steering_results[behaviour][layer] = {m: sum(layer_results[m]) / len(layer_results[m]) for m in multipliers}

    with open(f'results/{model.get_model_name()}_layer_sweep.json', 'w') as f:
        json.dump(steering_results, f, indent=2)
