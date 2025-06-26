import os
import json
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from main.loaders.llm_pure_text import LLMPureTextDataset
from main.loaders.chatglm_chat import ChatGLM_ChatDataset
from main.loaders.qwen_chat import QwenChatDataset
from main.loaders.llm_chat import LLMChatDataset
from main.loaders.llm_rlhf import LLM_RLHFDataset
from main.loaders.llm_dpo import LLMDPODataset
import torch

def collate_fn_wrapper(tokenizer):
    def left_pad_collate_fn(batch):
        result = {}
        max_length = 0
        max_length_without_last_turn = 0
        for item in batch:
            if item['input_ids'].shape[0] > max_length:
                max_length = item['input_ids'].shape[0]
            if 'input_ids_without_last_turn' in item and item['input_ids_without_last_turn'].shape[0] > max_length_without_last_turn:
                max_length_without_last_turn = item['input_ids_without_last_turn'].shape[0]
        for item in batch:
            for key in item:
                if key not in result:
                    result[key] = []
                if key in ('input_ids'):
                    pad_length = max_length - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([tokenizer.pad_token_id] * pad_length), item[key]], dim=-1)
                elif key == ('labels'):
                    pad_length = max_length - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([-100] * pad_length), item[key]], dim=-1)
                if key in ('input_ids_without_last_turn'):
                    pad_length = max_length_without_last_turn - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([tokenizer.pad_token_id] * pad_length), item[key]], dim=-1)
                elif key == ('labels_without_last_turn'):
                    pad_length = max_length_without_last_turn - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([-100] * pad_length), item[key]], dim=-1)
                result[key].append(item[key])
        for key in result:
            if key in ('input_ids', 'labels', 'input_ids_without_last_turn', 'labels_without_last_turn', 'last_input_len'):
                result[key] = torch.stack(result[key])
        return result
    return left_pad_collate_fn

def collate_fn_wrapper_dpo(tokenizer):
    def left_pad_collate_fn(batch):
        result = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "rejected_mask": [],
            "chosen_mask": [],
            "input_ids":[],
            "labels": []
        }
        max_length = 0
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) for item in batch)
            max_length = max(max_length, current_max)

        for item in batch:
            for key in item:
                if key == ("prompt"):
                    prompt = torch.tensor(item[key])
                    result[key].append(prompt)
                elif key in ("chosen", "rejected"):
                    pad_length = max_length - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([tokenizer.pad_token_id] * pad_length), item[key]], dim=-1)
                    mask = torch.ones(len(item[key])).bool()
                    mask_length = pad_length + prompt.shape[0]
                    mask[:mask_length] = False
                    result[key].append(item[key])
                    result[f"{key}_mask"].append(mask)
                elif key == ("input_ids"):
                    pad_length = max_length - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([tokenizer.pad_token_id] * pad_length), item[key]], dim=-1)
                    result[key].append(item[key])
                elif key == ("labels"):
                    pad_length = max_length - len(item[key])
                    item[key] = torch.cat([torch.LongTensor([-100] * pad_length), item[key]], dim=-1)
                    result[key].append(item[key])
                
        # prompt字段没有进行对齐
        for key in result:
            if key in ('chosen', 'rejected', 'chosen_mask', 'rejected_mask', 'input_ids', 'labels'):
                result[key] = torch.stack(result[key])
        return result
    return left_pad_collate_fn


class AutoDataloader():

    '''
    loader_name: str; the dataloader name
    data_path: str or obj; the path of the data; if str, it will use the present dataset in data_present_path, or you should define the path like e.g. { 'train': './train.json', 'dev': './dev.json' }
    model_type: interactive or siamese
    data_present_path: str; the path of the data_present; the data_present is a json file which contains the path of the dataset, and the format is like e.g. { 'dataset_name': {'train': './train.json', 'dev': './dev.json'} }
    max_length: int; the length of the padding
    '''

    def __init__(self, tokenizer, config, loader_name='LLM_Chat', data_path="Boss", data_present_path="/root/autodl-tmp/datasets/present.json", max_length=50):
        self.tokenizer = tokenizer
        self.loader_name = loader_name
        self.max_length = max_length
        self.data_present = self.get_data_present(data_present_path)
        self.data_path = self.data_present[data_path] if data_path in self.data_present else data_path
        if loader_name == 'LLM_Pure':
            self.train_set = LLMPureTextDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = LLMPureTextDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = LLMPureTextDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'ChatGLM_Chat':
            self.train_set = ChatGLM_ChatDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = ChatGLM_ChatDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = ChatGLM_ChatDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'Qwen_Chat':
            self.train_set = QwenChatDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = QwenChatDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = QwenChatDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'LLM_Chat':
            self.train_set = LLMChatDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = LLMChatDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = LLMChatDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'LLM_RLHF':
            self.train_set = LLM_RLHFDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = LLM_RLHFDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = LLM_RLHFDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'LLM_DPO':
            self.train_set = LLMDPODataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = LLMDPODataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = LLMDPODataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def __call__(self, batch_size=1, batch_size_eval=1, eval_mode='dev', use_collate=False):
        if not use_collate:
            dataiter = DataLoader(self.train_set, batch_size=batch_size)
            if eval_mode == 'dev':
                dataiter_eval = DataLoader(
                    self.eval_set, batch_size=batch_size_eval)
            else:
                dataiter_eval = DataLoader(
                    self.test_set, batch_size=batch_size_eval)
        else:
            # dpo特殊处理对齐
            if (self.loader_name == 'LLM_DPO'):
                left_pad_collate_fn = collate_fn_wrapper_dpo(self.tokenizer)
            else:
                left_pad_collate_fn = collate_fn_wrapper(self.tokenizer)
            dataiter = DataLoader(self.train_set, batch_size=batch_size, collate_fn=left_pad_collate_fn)
            if eval_mode == 'dev':
                dataiter_eval = DataLoader(
                    self.eval_set, batch_size=batch_size_eval, collate_fn=left_pad_collate_fn)
            else:
                dataiter_eval = DataLoader(
                    self.test_set, batch_size=batch_size_eval, collate_fn=left_pad_collate_fn)
        return dataiter, dataiter_eval
