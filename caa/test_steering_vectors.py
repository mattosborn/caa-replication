import torch as t
import json
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset, behaviours

def test_steering_vectors(model_name: str):
    model = ModelWrapper(model_name)
    multipliers = [-1, -0.5, 0, 0.5, 1]

    steering_results = {}

    for behaviour in behaviours:
        results = {m: [] for m in multipliers}
        dataset = load_dataset(f"{behaviour}_test_ab")
        steering_vectors = t.load(f'data/{behaviour}_steering_vectors.pt')

        for data in tqdm(dataset):
            prompt = model.tokenize_question(data['question'], '(')
            for m in multipliers:
                steered_output = model.prompt_with_steering(prompt, steering_dict={13: steering_vectors[13] * m})
                prob_behaviour = model.calc_behaviour_prob(steered_output.logits, data)
                results[m].append(prob_behaviour)

        steering_results[behaviour] = {m: sum(results[m]) / len(results[m]) for m in multipliers}

    with open(f'results/{model.get_model_name()}_steering_results.json', 'w') as f:
        json.dump(steering_results, f, indent=2)
