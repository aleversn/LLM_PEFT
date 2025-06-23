import os
import json
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PretrainedConfig, PreTrainedTokenizer


class LLM_RLHFDataset(Dataset):

    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer, config, file_name, max_length=512, do_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.data = self.load_jsonl(file_name)
        self.random_list = [idx for idx in range(len(self.data))]
        # 将预处理后的字段添加到列表中
        self.preprocess_list = [self.process_item(item) for item in self.data]
        if self.do_shuffle:
            random.shuffle(self.random_list)

    def load_jsonl(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return data

    def init_ids_and_masks(self):
        # 对qwen2/qwen2.5,不需要在开头填充特殊字符
        if self.config.model_type == 'qwen2':
            return [], []
        elif self.config.model_type == 'chatglm':
            # 新版tokenizer已不支持get_command，换用convert_tokens_to_ids获取特殊标记id
            # GLM3中标记文本开始的符号为sop，GLM4中为<sop>
            return [
                self.tokenizer.convert_tokens_to_ids('[gMASK]'),
                self.tokenizer.convert_tokens_to_ids('sop') if 'sop' in self.tokenizer.added_tokens_encoder else self.tokenizer.convert_tokens_to_ids('<sop>'),
            ], [False, False]
        else:
            raise NotImplementedError(f"{self.config.model_type} series loaders haven't been implemented yet.")
    
    def build_single_message(self, role, content):
        if self.config.model_type == 'chatglm':
            input_ids = self.tokenizer.build_single_message(role, '', content)
            role_ids = self.tokenizer.build_single_message(role, '', '')
            content_len = len(input_ids) - len(role_ids)
            return input_ids, content_len
        # qwen2/2.5中一条对话的格式要求:开始符+正文内容+结束符+回车符，示例:
        # <|im_strart|>user\n
        # Hello, what's your name?<|im_end|>\n
        elif self.config.model_type == 'qwen2':
            im_start_token = [self.tokenizer.convert_tokens_to_ids('<|im_start|>')]
            im_end_token = [self.tokenizer.convert_tokens_to_ids('<|im_end|>')]
            nl_token = self.tokenizer.encode('\n')
            input_ids = im_start_token + self.tokenizer.encode('role') + nl_token + \
                        self.tokenizer.encode(content) + im_end_token + nl_token
            return input_ids, len(self.tokenizer.encode(content))
        # 如有需要，可以参照相应模型的对话格式要求，自行实现新模型的loader
        else:
            raise NotImplementedError(f"{self.config.model_type} series loaders haven't been implemented yet.")
             

    def process_item(self, item):
        conv = item['conversations'] if 'conversations' in item else item    
        input_ids, loss_masks = self.init_ids_and_masks()

        # RLHF的新加字段
        last_input_len = 0
        last_user_content = ''
        last_assistant_content = ''

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids, last_input_len = self.build_single_message(message['role'], message['content'])
                new_loss_masks = [loss_mask_val] * len(new_input_ids)
                # 为新加的字段赋值
                if message['role'] == 'user':
                    last_user_content = message['content']
                    last_input_len = 0 # 这里可能会有问题，尽量确保你喂入的数据最后以assistant结尾
                elif message['role'] == 'assistant':
                    last_assistant_content = message['content']

            input_ids += new_input_ids
            loss_masks += new_loss_masks
        
        # 此处要注意一下各模型的对话格式，有的模型(如glm3, glm4)在一组对话完毕后可能会有结束符需要填充
        if self.config.model_type == 'chatglm':
            input_ids.append(self.tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = self.max_length

        # RLHF新加入的字段
        input_ids_without_last_turn = input_ids[:-last_input_len]# 针对多轮会话，只取到最后一个问题为止；注意如果后续需要做在线RL，这里需要考虑进行修改
        labels_without_last_turn = labels[:-last_input_len]

        # 对超长的字符进行截断
        exceed_len = len(input_ids) - max_length
        if exceed_len > 0:
            last_input_len = last_input_len - exceed_len
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        input_ids_without_last_turn = input_ids_without_last_turn[:max_length]
        labels_without_last_turn = labels_without_last_turn[:max_length]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_ids_without_last_turn': input_ids_without_last_turn,
            'labels_without_last_turn': labels_without_last_turn,
            'last_input_len': last_input_len,
            'last_user_content': last_user_content,
            'last_assistant_content': last_assistant_content}

    def __getitem__(self, index):
        index = self.random_list[index]
        item = self.data[index]
        input_ids, labels, input_ids_without_last_turn, labels_without_last_turn, last_input_len, last_user_content, last_assistant_content = self.preprocess_list[index].values(
        )
        gold_answers = item['gold_answers']
        bad_answers = item['bad_answers']

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        input_ids_without_last_turn = torch.tensor(input_ids_without_last_turn)
        labels_without_last_turn = torch.tensor(labels_without_last_turn)

        return {
            'query': last_user_content,
            'gold_answers': gold_answers,
            'bad_answers': bad_answers,
            'input_ids': input_ids,
            'input_ids_without_last_turn': input_ids_without_last_turn,
            'labels_without_last_turn': labels_without_last_turn,
            'last_input_len': torch.tensor(last_input_len),
            'last_assistant_content': last_assistant_content,
            'labels': labels
        }

    def __len__(self):
        return len(self.data)
