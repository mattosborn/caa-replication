import torch as t
import json
from tqdm import tqdm, trange
from caa.model import ModelWrapper
from caa.utils import load_dataset, behaviours, BatchedDataset, write_results

def test_steering_vectors(model_name: str):
    model = ModelWrapper(model_name)
    multipliers = [-1, -0.5, 0, 0.5, 1]

    for behaviour in tqdm(behaviours, desc='behaviour'):
        dataset = load_dataset(f"{behaviour}_test_ab")
        steering_vectors = t.load(f'data/{behaviour}_steering_vectors.pt')
        for layer in trange(0, 32, desc='layer'):
            for m in tqdm(multipliers, desc='multiplier'):
                raw_results = []
                for batch in BatchedDataset(dataset, 32):
                    output = model.batch_prompt(batch, steering_dict={layer: steering_vectors[layer] * m})
                    raw_results += [
                        q | {
                            'answer_probabilities': output['answer_probabilities'][i],
                            'prob_matching_behaviour': output['probabilities'][i]
                        }
                        for i, q in enumerate(batch)
                    ]
                write_results({'behaviour': behaviour, 'multiplier': m, 'layer': layer, 'model': model.get_model_name()}, raw_results)
