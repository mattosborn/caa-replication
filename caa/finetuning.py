import torch as t
import json
from tqdm import tqdm
from caa.model import ModelWrapper
from caa.utils import load_dataset, behaviours, layers, BatchedDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def finetune(model_name: str):
    model = ModelWrapper(model_name)

    dataset = load_dataset("coordinate")
    tokenized_dataset = []

    for prompt in dataset:
        tokenized_dataset.append(model.tokenize_question(prompt["question"], prompt["answer_matching_behaviour"]))


    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    def train(model_name, epochs):


if __name__ == "__main__":
    finetune("meta-llama/Llama-2-7b-chat-hf")