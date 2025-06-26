import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer

class LLMChatDataset(Dataset):

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
    
    def init_ids_and_masks(self):
        # Llama 2
        if "<|begin_of_text|>" not in self.tokenizer.special_tokens_map.values():
            return [], []
        # Llama 3
        else:
            return [self.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')], [False]

    def build_single_message(self, t):
        ids = self.tokenizer.apply_chat_template([t])
        if t['role'] in ['user', 'system']:
            ls = [-100 for _ in ids]
            return ids, ls
        
        # 由于只需要训练生成的回答，因此要mask掉最后一组对话的身份信息以及无关的符号
        # Llama 2
        if "<|begin_of_text|>" not in self.tokenizer.special_tokens_map.values():
            tokens_to_train = self.tokenizer.encode(t['content']) + [self.tokenizer.convert_tokens_to_ids('</s>')]
        # Llama 3
        else:
            tokens_to_train = self.tokenizer.encode(t['content']) + [self.tokenizer.convert_tokens_to_ids('<|eot_id|>')]

        ls = [-100] * (len(ids) - len(tokens_to_train)) + tokens_to_train
        return ids, ls

    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item

        input_ids, labels = self.init_ids_and_masks()

        for t in conv:
            ids, ls = self.build_single_message(t)

            input_ids.extend(ids)
            labels.extend(ls)
            
        max_length = self.max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        return {'input_ids': input_ids, 'labels': labels}
    
    def __getitem__(self, index):
        index = self.random_list[index]
        data = self.data[index]
        input_ids, labels = self.process_item(data).values()

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data)