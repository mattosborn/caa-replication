import os
from tqdm import tqdm
from .model import ModelWrapper
from .utils import load_dataset, behaviours


def test_baseline(model_name: str):
    model = ModelWrapper(model_name, os.getenv('HF_TOKEN'))

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
            baseline_answer = model.tokenizer.decode(
                baseline.logits[0, -1].argmax())
            is_matching = baseline_answer == data['answer_matching_behavior'][1]
            answers.append(is_matching)

        answers_matching_behaviour = len(
            [a for a in answers if a == True])
        result = answers_matching_behaviour / len(answers)
        print(
            f'{behaviour}: {answers_matching_behaviour}/{len(answers)} ({result*100:.2f}%)')
