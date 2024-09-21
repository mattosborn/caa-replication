import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import re
from typing import Dict, List
from dotenv import load_dotenv
import os
from .utils import QuestionData, SteeringDict

from typing import TypedDict

load_dotenv()
    
class ModelWrapper:
    def __init__(self, model_name, hf_token=os.getenv('HF_TOKEN')):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, device_map="cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, device_map="cuda:0")
        print("Model device", self.model.device)
        
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = "[PAD]"
        self.last_activations: Dict[str, t.Tensor] = {} # layer name -> activations
        self.activation_handles = []
        

    def setup_layer_activation_hooks(self):
        self.last_activations = {}
        
        def create_hook(layer_name):
            def hook(m, i, o): return self.last_activations.update(
                    {layer_name: o[0]})
            return hook
        
        for name, layer in self.model.named_modules():
            if re.search(r"^model\.layers\.(\d\d?)$", name):
                handle = layer.register_forward_hook(create_hook(name))
                self.activation_handles.append(handle)

    def remove_layer_activation_hooks(self):
        for handle in self.activation_handles:
            handle.remove()
        self.activation_handles = []
        self.last_activations = {}

    def __call__(self, prompt) -> CausalLMOutputWithPast:
        _prompt = prompt.to(self.model.device)
        with t.no_grad():
            return self.model(_prompt)
    
    def get_last_layer_activations(self):
        if not self.last_activations: raise ValueError("run setup_layer_activation_hooks() first")
        return [self.last_activations[f'model.layers.{i}'] for i in range(len(self.model.model.layers))]
    
    
    def calc_behaviour_prob(self, logits: t.Tensor, data: QuestionData):
        probabilities = t.softmax(logits, -1)
        answer_char = data['answer_matching_behavior'][1]
        answer_chars = re.findall(r'\(([A-Z])\)', data['question'])
        answer_tokens = [self.tokenizer.convert_tokens_to_ids(a) for a in answer_chars]
        answer_probabilities = [probabilities[token].item() for token in answer_tokens]
        behaviour_token = self.tokenizer.convert_tokens_to_ids(answer_char)
        prob_behaviour = probabilities[behaviour_token].item()
        return prob_behaviour / sum(answer_probabilities)
    
    def create_prompt_str(self, question, answer=None):
        if re.search(r"Llama-2.*\bchat\b", self.model_name):
            # tokenize for llama-2 chat models
            prompt = f"[INST] {question.strip()} [/INST]"
            if answer:
                prompt += f" {answer.strip()}"    
            return prompt
        raise ValueError(f"Model {self.model_name} not supported")
    
    def batch_prompt(self, data: List[QuestionData], steering_dict: SteeringDict = {}):
        prompt_strs = [self.create_prompt_str(d['question'], '(') for d in data]
        lengths: List[int] = self.tokenizer(prompt_strs, return_length=True)['length'] # type: ignore
        input_tokens = self.tokenizer(prompt_strs, padding='longest', return_tensors='pt')['input_ids']
        
        # register steering hooks
        hook_handles = []
        for layer, steering in steering_dict.items():
            def hook(m, i, output):
                for i, length in enumerate(lengths):
                    output[0][i, length - 1, :] += steering.to(output[0].device)
                return output
                
            handle = self.model.model.layers[layer].register_forward_hook(hook)
            hook_handles.append(handle)

        output = self(input_tokens)

        # remove steering hooks
        for handle in hook_handles:
            handle.remove()
        
        ModelOutputWithProbabilities = TypedDict('ModelOutputWithProbabilities', {'probabilities': List[float], 'logits': List[t.Tensor]})
        prompt_result: ModelOutputWithProbabilities = {
            'probabilities': [
                self.calc_behaviour_prob(output.logits[i, lengths[i] - 1], d) for i, d in enumerate(data)
            ],
            'logits': [output.logits[i] for i in range(len(data))]
        }
        
        return prompt_result
    
    def batch_activation_prompt(self, data: List[QuestionData]):
        
        prompts_pos = [self.create_prompt_str(d['question'], d['answer_matching_behavior'].strip()[:2]) for d in data]
        prompts_neg = [self.create_prompt_str(d['question'], d['answer_not_matching_behavior'].strip()[:2]) for d in data]
        
        all_prompts = prompts_pos + prompts_neg
        
        lengths: List[int] = self.tokenizer(all_prompts, return_length=True)['length'] # type: ignore
        input_tokens = self.tokenizer(all_prompts, padding='longest', return_tensors='pt')['input_ids']
        
        output = self(input_tokens)
        
        layer_activations = self.get_last_layer_activations()
        
        get_pos_activation = lambda layer_num, i: layer_activations[layer_num][i, lengths[i] - 1].detach()
        get_neg_activation = lambda layer_num, i: layer_activations[layer_num][i + len(data), lengths[i + len(data)] - 1].detach()
        
        get_diff = lambda layer_num, i: get_pos_activation(layer_num, i) - get_neg_activation(layer_num, i)
        num_questions = len(data)
        num_layers = len(layer_activations)
        
        ModelOutputWithActivations = TypedDict('ModelOutputWithActivations', {'diffs': t.Tensor, 'logits': List[t.Tensor]})
        prompt_result: ModelOutputWithActivations = {
            'logits': [output.logits[i] for i in range(len(data))],
            'diffs': t.stack([t.stack([get_diff(j, i).to('cpu') for j in range(num_layers)]) for i in range(num_questions)])
        }
        
        return prompt_result
        
    def tokenize_question(self, question, answer=None):
        prompt = self.create_prompt_str(question, answer)
        tokens = self.tokenizer.encode(prompt)
        return t.tensor(tokens)
        
    def get_model_name(self):
        return self.model_name.replace('/', '-')