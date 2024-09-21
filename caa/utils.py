import json

def load_dataset(name):
    dataset_file_name = f'datasets/{name}.json'
    dataset_file = open(dataset_file_name, 'r')
    dataset = json.load(dataset_file)
    return dataset


# behaviours = ['coordinate', 'corrigible', 'hallucination',
#               'myopic', 'refusal', 'survival', 'sycophancy']

behaviours = ["coordinate", "corrigible", "myopic", "refusal", "survival"]

layers = [i for i in range(32)]
