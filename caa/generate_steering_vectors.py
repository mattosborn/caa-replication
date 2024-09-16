import torch as t
import os
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset ##, behaviours

# behaviours = ['myopic', 'refusal', 'survival', 'sycophancy'] # 'hallucination', 
behaviours = ['sycophancy', 'hallucination']

def generate_steering_vectors(model_name: str):
    model = ModelWrapper(model_name, os.getenv('HF_TOKEN'))
    model.setup_layer_activation_hooks()
    
    for behaviour in behaviours:
        dataset = load_dataset(behaviour)

        # the model returns a dictionary of activations in `Dict[string, Tensor]` format
        # this stacks it into a single tensor with the activations for the last token in each layer
        def process_activations(activations): return t.stack(
            [activation.to('cpu')[:, -1] for activation in activations], dim=1)

        all_activations_p = []
        all_activations_n = []

        # TODO: we probably want to batch this as some of the datasets are large
        for data in tqdm(dataset):
            input_p = model.tokenize_question(
                data['question'], data['answer_matching_behavior'].strip()[:2])
            
            input_n = model.tokenize_question(
                data['question'], data['answer_not_matching_behavior'].strip()[:2])

            model(input_p)
            activations_p = model.get_last_layer_activations()
            all_activations_p.append(process_activations(activations_p))

            model(input_n)
            activations_n = model.get_last_layer_activations()
            all_activations_n.append(process_activations(activations_n))

        # stack all tensors and flatten batch and question dimensions
        activations_p_tensor = t.stack(all_activations_p, dim=0).flatten(0, 1)
        activations_n_tensor = t.stack(all_activations_n, dim=0).flatten(0, 1)

        diffs = activations_p_tensor - activations_n_tensor

        steering_vectors = diffs.mean(0)

        with open(f'data/{behaviour}_steering_vectors.pt', 'wb') as f:
            t.save(steering_vectors, f)

    model.remove_layer_activation_hooks()