import torch as t
import os
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset, BatchedDataset ##, behaviours

behaviours = ['myopic', 'refusal', 'survival', 'corrigible', 'coordinate']
# behaviours = ['sycophancy', 'hallucination']

def generate_steering_vectors(model_name: str):
    model = ModelWrapper(model_name)
    
    model.setup_layer_activation_hooks()
    
    for behaviour in behaviours:
        dataset = load_dataset(behaviour)
        print(f"Generating steering vectors for {behaviour}")

        all_diffs = []

        for batch in tqdm(BatchedDataset(dataset, 4)):
            output = model.batch_activation_prompt(batch)
            all_diffs.extend(output['diffs'])

        print(t.stack(all_diffs, dim=0).shape)
        steering_vectors = t.stack(all_diffs, dim=0).mean(dim=0)

        with open(f'data/{behaviour}_steering_vectors.pt', 'wb') as f:
            t.save(steering_vectors, f)

    model.remove_layer_activation_hooks()