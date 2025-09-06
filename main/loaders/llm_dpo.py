import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer

class LLMDPODataset(Dataset):

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    
    def __init__(self, tokenizer, config, file_name, max_length=512, do_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.data = self.load_jsonl(file_name)
        self.random_list = [idx for idx in range(len(self.data))]
        if self.do_shuffle:
            random.shuffle(self.random_list)
            
    
    def load_jsonl(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return data
    
    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item

        input_ids, labels = [], []
        for t in conv:
            role = t['role']
            ids = self.tokenizer.apply_chat_template([t])
            ls = ids if role not in ['user', 'system'] else [-100 for _ in ids]
            input_ids.extend(ids)
            labels.extend(ls)
            
        chosen = item['gold_answers']
        rejected = item['bad_answers']
        # 对应[gMASK] <sop> <|assistant|> \n
        ids1 = [151331, 151333, 151337, 198]
        ids1.extend(self.tokenizer.encode(chosen, add_special_tokens=False))
        chosen_full_tokens = []
        chosen_full_tokens.extend(prompt)
        chosen_full_tokens.extend(ids1)
        # 对应[gMASK] <sop> <|assistant|> \n
        ids2 = [151331, 151333, 151337, 198]
        ids2.extend(self.tokenizer.encode(rejected, add_special_tokens=False))
        rejected_full_tokens = []
        rejected_full_tokens.extend(prompt)
        rejected_full_tokens.extend(ids2)

        max_length = self.max_length
        prompt = prompt[:max_length]
        chosen_full_tokens = chosen_full_tokens[:max_length]
        rejected_full_tokens = rejected_full_tokens[:max_length]
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        return {'prompt': prompt, 'chosen': chosen_full_tokens, 'rejected': rejected_full_tokens, 'input_ids': input_ids, 'labels': labels}
    
    def __getitem__(self, index):
        index = self.random_list[index]
        data = self.data[index]
        prompt, chosen, rejected, input_ids, labels = self.process_item(data).values()

        prompt = torch.tensor(prompt)
        chosen = torch.tensor(chosen)
        rejected = torch.tensor(rejected)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'input_ids': input_ids,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data)