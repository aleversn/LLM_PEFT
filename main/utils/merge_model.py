import os
import uuid
import torch
import re
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import jieba
from tqdm import tqdm
from main.loader import AutoDataloader
from main.analysis import Analysis
import copy
from accelerate import Accelerator
from main.models.DPOLoss import DPOLoss 
import json
accelerator = Accelerator()


class Trainer():

    def __init__(self, tokenizer, from_pretrained, config=None, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.from_pretrained = from_pretrained
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.eval_mode = eval_mode
        self.config_init()
        self.model_init(resume_path=resume_path)
        self.analysis = Analysis()


    def config_init(self):
        self.config = AutoConfig.from_pretrained(
            self.from_pretrained, trust_remote_code=True) if self.config is None else self.config
        if self.config.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def model_init(self, resume_path=None):
        if self.accelerate.is_local_main_process:
            print('AutoModel Choose Model: {}\n'.format(
                self.from_pretrained))
        if self.config.model_type == 'chatglm':
            target_modules=['query_key_value']
            self.model = AutoModel.from_pretrained(
                self.from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        elif self.config.model_type == 'llama':
            target_modules=["q_proj", "k_proj", "v_proj"]
            self.model = LlamaForCausalLM.from_pretrained(
                self.from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        elif self.config.model_type == 'qwen':
            target_modules=["c_attn", "c_proj", "w1", "w2"]
            self.model = AutoModelForCausalLM.from_pretrained(
                self.from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.config.model_type == 'qwen2':
            target_modules=["q_proj", "k_proj", "v_proj"]
            self.model = AutoModelForCausalLM.from_pretrained(
                self.from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            target_modules=target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(
                self.model, resume_path, config=peft_config)
            merged_model = self.model.merge_and_unload()  
            merged_model.save_pretrained("./save_model/resume_1000_nt2_15500_glm4")
            self.tokenizer.save_pretrained("./save_model/resume_1000_nt2_15500_glm4")
        else:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
