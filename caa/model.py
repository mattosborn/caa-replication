import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List
from dotenv import load_dotenv
import os

load_dotenv()

class ModelWrapper:
    def __init__(self, model_name, hf_token=os.getenv('HF_TOKEN')):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, device_map="cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, device_map="cuda:0")
        print("Model device", self.model.device)
        
        # self.tokenizer.add_special_tokens({'pad_token': ''})
        self.tokenizer.pad_token = "[PAD]"
        

    def setup_layer_activation_hooks(self):
        self.activation_handles = []
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
        self.activation_handles = None
        self.last_activations = None

    def __call__(self, prompt):
        _prompt = prompt.to(self.model.device)
        with t.no_grad():
            return self.model(_prompt)
    
    def get_last_layer_activations(self):
        if not hasattr(self, 'last_activations'):
            raise ValueError("run setup_layer_activation_hooks() first")
        return [self.last_activations[f'model.layers.{i}'] for i in range(len(self.model.model.layers))]
    
    def calc_behaviour_prob(self, logits, data, prompt_length=-1):
        probabilities = t.softmax(logits[prompt_length], -1)
        answer_char = data['answer_matching_behavior'][1]
        answer_chars = re.findall(r'\(([A-Z])\)', data['question'])
        answer_tokens = [self.tokenizer.convert_tokens_to_ids(a) for a in answer_chars]
        answer_probabilities = [probabilities[token].item() for token in answer_tokens]
        behaviour_token = self.tokenizer.convert_tokens_to_ids(answer_char)
        prob_behaviour = probabilities[behaviour_token].item()
        return prob_behaviour / sum(answer_probabilities)
    
    def calc_batch_probs(self, logits, batch_prompt, questions):
        return [self.calc_behaviour_prob(logits[i], data, batch_prompt['lengths'][i] - 1) for i, data in enumerate(questions)]
        
    def prompt_with_steering(self, batch_prompt, steering_dict: Dict[int, t.tensor]):
        hook_handles = []
        # register steering hooks
        for layer, steering in steering_dict.items():
            def hook(m, i, output):
                for batch_number, length in enumerate(batch_prompt['lengths']):
                    output[0][batch_number, length - 1, :] += steering.to(output[0].device)
                    return output
            handle = self.model.model.layers[layer].register_forward_hook(hook)
            hook_handles.append(handle)

        # invoke model
        logits = self(batch_prompt['prompts'])

        # remove steering hooks
        for handle in hook_handles:
            handle.remove()

        return logits

    def create_prompt_str(self, question, answer=None):
        if re.search(r"Llama-2.*\bchat\b", self.model_name):
            # tokenize for llama-2 chat models
            prompt = f"[INST] {question.strip()} [/INST]"
            if answer:
                prompt += f" {answer.strip()}"    
            return prompt
        raise ValueError(f"Model {self.model_name} not supported")

    def tokenize_question(self, question, answer=None):
        prompt = self.create_prompt_str(question, answer)
        tokens = self.tokenizer.encode(prompt)
        return t.tensor(tokens).unsqueeze(0)
        
    def tokenize_batch(self, prompt_strs):
        prompts = self.tokenizer(prompt_strs, padding='longest', return_tensors='pt')['input_ids']
        prompt_lengths = self.tokenizer(prompt_strs, return_length=True)['length']
        return { 'prompts': prompts, 'lengths': prompt_lengths }
    
    # def create_batched_prompt(self, List[t.tensor]):
        
    def get_model_name(self):
        return self.model_name.replace('/', '-')