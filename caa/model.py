import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict
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

    def __call__(self, *args, **kwds):
        with t.no_grad():
            return self.model(*args, **kwds)
    
    
    def get_last_layer_activations(self):
        if not hasattr(self, 'last_activations'):
            raise ValueError("run setup_layer_activation_hooks() first")
        return [self.last_activations[f'model.layers.{i}'] for i in range(len(self.model.model.layers))]

    def prompt_with_steering(self, input_tokens: t.tensor, steering_dict: Dict[int, t.tensor]):
        hook_handles = []

        # register steering hooks
        for layer, steering in steering_dict.items():
            def hook(m, i, output):
                output[0][:, -1, :] += steering.to(output[0].device)
                return output
            handle = self.model.model.layers[layer].register_forward_hook(hook)
            hook_handles.append(handle)

        # invoke model
        activations = self(input_tokens)

        # remove steering hooks
        for handle in hook_handles:
            handle.remove()

        return activations

    def tokenize_question(self, question, answer=None):
        if re.search(r"Llama-2.*\bchat\b", self.model_name):
            # tokenize for llama-2 chat models
            prompt = f"[INST] {question.strip()} [/INST]"
            if answer:
                prompt += f" {answer.strip()}"
            tokens = self.tokenizer.encode(prompt)
            return t.tensor(tokens).unsqueeze(0)
        
        raise ValueError(f"Model {self.model_name} not supported")
