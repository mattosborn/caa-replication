from tqdm import tqdm
from .model import ModelWrapper
from .utils import load_dataset, behaviours
from statistics import mean

device = "cuda:0"

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
            prompt = model.tokenize_question(data['question'], '(').to(device)
            baseline = model(prompt)
            prob_behaviour = model.calc_behaviour_prob(baseline.logits, data)
            answers.append(prob_behaviour)

        mean_answers = mean(answers)
        print(f'{behaviour}: {mean_answers:.4f}')
        
