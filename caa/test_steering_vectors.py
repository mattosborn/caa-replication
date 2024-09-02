import torch as t
import os
from tqdm import tqdm
from .model import ModelWrapper
from .utils import load_dataset


def test_steering_vectors(model_name: str):
    model = ModelWrapper(model_name, os.getenv('HF_TOKEN'))

    steering_vectors = t.load('data/steering_vectors.pt')
    baseline_answers = []
    steered_answers = []

    test_dataset = load_dataset('refusal_test_ab')

    for data in test_dataset[:2]:
        print(f"\n-------------\n\n{data['question']}\n")

        prompt = model.tokenize_question(data['question'], '(')
        baseline = model(prompt)

        indices = baseline.logits[0, -1].topk(5).indices
        probabilities = t.softmax(baseline.logits[0, -1], -1)
        print("BASELINE:")
        for i in range(5):
            print(
                f"{i}: {model.tokenizer.decode(indices[i])} ({probabilities[indices[i]]})")

        steered_output = model.prompt_with_steering(
            prompt, steering_dict={13: steering_vectors[13]})

        indices = steered_output.logits[0, -1].topk(5).indices
        probabilities = t.softmax(steered_output.logits[0, -1], -1)
        print("STEERED:")
        for i in range(5):
            print(
                f"{i}: {model.tokenizer.decode(indices[i])} ({probabilities[indices[i]]})")
