import torch as t
import os
from tqdm import tqdm
from .model import ModelWrapper
from .utils import load_dataset, behaviours
from statistics import mean

def test_baseline(model_name: str):
    model = ModelWrapper(model_name)
    
    for behaviour in behaviours:
        test_dataset = load_dataset(f'{behaviour}_test_ab')

        # TODO: get token probabilities for the matching behaviour
        #       and avg them over all questions to get the baseline
        # from the paper (p6, table 3):
        #   Scores are average token probabilities given to answer matching behavior over the 50 test examples.

        answers = []

        for data in tqdm(test_dataset, leave=False):
            prompt = model.tokenize_question(data['question'], '(')
            baseline = model(prompt)
            
            probabilities = t.softmax(baseline.logits[0, -1], -1)
            
            answer_char = data['answer_matching_behavior'][1]
            behaviour_token = model.tokenizer.convert_tokens_to_ids(answer_char)
            
            answers.append(probabilities[behaviour_token].item())

        mean_answers = mean(answers)
        print(f'{behaviour}: {mean_answers:.4f}')
        
