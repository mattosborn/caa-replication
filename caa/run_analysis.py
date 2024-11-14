from caa.utils import read_results, behaviours, get_model_filename
import json

def aggregate_steering_results(model_name: str):
  ## aggregate by behaviour -> layer -> multiplier
  steering_results = {}
  for behaviour in behaviours:
    steering_results[behaviour] = {}
    for layer in range(32):
      steering_results[behaviour][layer] = {}
      for m in [-1, -0.5, 0, 0.5, 1]:
        results = read_results({'behaviour': behaviour, 'model': model_name, 'layer': layer, 'multiplier': m})
        probabilities = [r['prob_matching_behaviour'] for r in results]
        steering_results[behaviour][layer][m] = sum(probabilities) / len(probabilities)
  with open(f'results/steering_{get_model_filename(model_name)}.json', 'w') as f:
    json.dump(steering_results, f, indent=2)
  
def run_analysis(model_name: str):
  aggregate_steering_results(model_name)
