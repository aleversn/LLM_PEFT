import os
import json
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from CC.loaders.chatglm_std import ChatGLM_LoRADataset
from CC.loaders.chatglm_chat import ChatGLM_ChatDataset


class AutoDataloader():

    '''
    loader_name: str; the dataloader name
    data_path: str or obj; the path of the data; if str, it will use the present dataset in data_present_path, or you should define the path like e.g. { 'train': './train.json', 'dev': './dev.json' }
    model_type: interactive or siamese
    data_present_path: str; the path of the data_present; the data_present is a json file which contains the path of the dataset, and the format is like e.g. { 'dataset_name': {'train': './train.json', 'dev': './dev.json'} }
    max_length: int; the length of the padding
    '''

    def __init__(self, tokenizer, config, loader_name='ChatGLM_LoRA', data_path="Boss", data_present_path="./data/present.json", max_length=50):
        self.loader_name = loader_name
        self.max_length = max_length
        self.data_present = self.get_data_present(data_present_path)
        self.data_path = self.data_present[data_path] if data_path in self.data_present else data_path
        if loader_name == 'ChatGLM_LoRA':
            self.train_set = ChatGLM_LoRADataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = ChatGLM_LoRADataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = ChatGLM_LoRADataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)
        elif loader_name == 'ChatGLM_Chat':
            self.train_set = ChatGLM_ChatDataset(
                tokenizer, config, self.data_path['train'], max_length=self.max_length, do_shuffle=True)
            self.eval_set = ChatGLM_ChatDataset(
                tokenizer, config, self.data_path['dev'], max_length=self.max_length, do_shuffle=False)
            if 'test' in self.data_path:
                self.test_set = ChatGLM_ChatDataset(
                    tokenizer, config, self.data_path['test'], max_length=self.max_length, do_shuffle=False)

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

    def __call__(self, batch_size=1, batch_size_eval=1, eval_mode='dev'):
        dataiter = DataLoader(self.train_set, batch_size=batch_size)
        if eval_mode == 'dev':
            dataiter_eval = DataLoader(
                self.eval_set, batch_size=batch_size_eval)
        else:
            dataiter_eval = DataLoader(
                self.test_set, batch_size=batch_size_eval)
        return dataiter, dataiter_eval
