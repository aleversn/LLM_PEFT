import os
import re
import gc
import uuid
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from main.models.chatglm import CCGPTModel
from transformers import AutoTokenizer, AutoModel


class GPTPredict():

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 model_name: str = "ChatGLM2-6B",
                 resume_path: str = None
                 ):
        '''
        GPTPredict: GPT预测器 (GPT predictor)

        ### Args:

        `num_gpus`: 使用的GPU编号列表 (the list of GPU numbers)

        `model_config_file_name`: bert配置文件名 (bert config file name)

        `model_name`: 模型名称 (the name of the model)

        `resume_path`: 恢复模型路径 (resume model path)
        '''
        self.model_name = model_name
        self.model_from_pretrained = model_from_pretrained
        self.model_init(model_name=model_name,
                        model_from_pretrained=model_from_pretrained)
        if resume_path is not None:
            self.model_to_device(resume_path=resume_path, gpu=num_gpus)

    def model_init(self, model_name,
                   model_from_pretrained):
        '''
        model_init: 模型初始化 (model initialization)

        ### Args:

        `model_name`: 模型名称 (the name of the model)

        `model_from_pretrained`: 从预训练模型中加载模型 (load model from pretrained model)
        '''
        ccModel = CCGPTModel(model_name,
                             model_from_pretrained=model_from_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.model = ccModel()
        self.model = self.model.eval()

    def model_to_device(self, resume_path=None, gpu=[0]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()

        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(
                resume_path))
            bert_model_dict = torch.load(resume_path).module.state_dict()
            self.model.module.load_state_dict(bert_model_dict)
        self.model.to(self.device)

    def predict(self, sentence, history=[]):
        with torch.no_grad():
            res, his = self.model.chat(
                self.tokenizer, sentence, history=history)
        return {
            "data": res,
            "history": his
        }

    def generate(self, text, max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8):
        with torch.no_grad():
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature}
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = inputs.to(self.device)
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.get_command("<|user|>"),
                            self.tokenizer.get_command("<|observation|>")]
            outputs = self.model.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
            response = self.tokenizer.decode(outputs)
        return response

    def predict_stream(self, sentence, history=[]):
        with torch.no_grad():
            for res, his in self.model.stream_chat(self.tokenizer, sentence, history=history):
                yield {
                    "data": res,
                    "history": his
                }

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX

    def __call__(self, sentence, history=[]):
        return self.predict(sentence, history=history)
