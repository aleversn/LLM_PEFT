import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm
from main.loader import AutoDataloader
from main.analysis import Analysis
import copy
from accelerate import Accelerator
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from main.models.DPOLoss import DPOLoss 
from accelerate import cpu_offload_with_hook
import json
accelerator = Accelerator()
import os
import csv
from transformers import TrainerCallback

class CSVLoggerCallback(TrainerCallback):
    def __init__(self,
                 train_csv_path="train_metrics.csv",
                 eval_csv_path="eval_metrics.csv"):
        # 记录路径
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path

        # 确保目录存在
        os.makedirs(os.path.dirname(train_csv_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(eval_csv_path) or ".", exist_ok=True)

        # 打开文件，检查是否需要写 header
        for path in (train_csv_path, eval_csv_path):
            if not os.path.exists(path):
                # Will create file and write header later
                open(path, "w").close()

    def _write_row(self, path, keys, values):
        write_header = os.path.getsize(path) == 0
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(keys)
            writer.writerow(values)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: 
            return
        # 区分 train 和 eval
        train_logs = {k:v for k,v in logs.items() if not k.startswith("eval_")}
        eval_logs  = {k[5:]:v for k,v in logs.items() if k.startswith("eval_")}  # 去掉前缀
        
        # 写 train
        if train_logs:
            keys   = ["step"] + sorted(train_logs.keys())
            values = [state.global_step] + [train_logs[k] for k in sorted(train_logs.keys())]
            self._write_row(self.train_csv_path, keys, values)
        
        # 写 eval
        if eval_logs:
            keys   = ["step"] + sorted(eval_logs.keys())
            values = [state.global_step] + [eval_logs[k] for k in sorted(eval_logs.keys())]
            self._write_row(self.eval_csv_path, keys, values)

class Trainer():

    def __init__(self, tokenizer, from_pretrained, data_path, data_present_path, config=None, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.from_pretrained = from_pretrained
        self.data_present = self.get_data_present(data_present_path)
        self.data_path = self.data_present[data_path] if data_path in self.data_present else data_path
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.eval_mode = eval_mode
        self.config_init()
        self.dataloader_init()
        self.model_init(resume_path=resume_path)
        self.analysis = Analysis()

    def get_data_present(self, present_path):
        if not os.path.exists(present_path):
            return {}
        with open(present_path, encoding='utf-8') as f:
            present_json = f.read()
        data_present = json.loads(present_json)
        return data_present

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
                self.model, resume_path, config=peft_config, is_trainable=True)
        else:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()


    def dataloader_init(self):
        self.datasets = load_dataset(
            "json",
            data_files={
                "train": self.data_path['train'],
                "dev": self.data_path['dev']
            },
            split=None    
        )

    def __call__(self, reward_func=[], lr=5e-6, weight_decay=0.1, warmup_ratio=0.2, logging_steps=20, gradient_accumulation_steps=2, max_completion_length=500, per_device_train_batch_size = 8,\
                 num_generations = 4, num_train_epochs=30, fp16=True, use_vllm=True, save_strategy="epoch", eval_strategy="epoch", logging_strategy="epoch", report_to=["tensorboard"]):
        return self.train(reward_func=reward_func, lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio, logging_steps=logging_steps, gradient_accumulation_steps=gradient_accumulation_steps,\
                          max_completion_length=max_completion_length, per_device_train_batch_size = per_device_train_batch_size, num_generations = num_generations, num_train_epochs=num_train_epochs, \
                            fp16=fp16, use_vllm=use_vllm, save_strategy=save_strategy, eval_strategy=eval_strategy, logging_strategy=logging_strategy, report_to=report_to)
    
    
    def train(self, reward_func, lr, weight_decay, warmup_ratio, logging_steps, gradient_accumulation_steps,\
              max_completion_length, per_device_train_batch_size, num_generations, num_train_epochs, \
                fp16, use_vllm, save_strategy, eval_strategy, logging_strategy, report_to):
        
        
        training_args = GRPOConfig(
            output_dir=f'./save_model/{self.task_name}',
            learning_rate=lr,
            #beta=0.001,                  # 控制策略偏离参考模型的程度，值越大更新越保守，默认 0.04  
            weight_decay=weight_decay,             # 权重衰减，通常用于防止过拟合
            warmup_ratio=warmup_ratio,             # 学习率 warmup 比例
            logging_steps=logging_steps,             # 一个batch更新一次，每经过 20 次参数更新就会打印一次训练指标
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=max_completion_length,    # 每个 prompt 后生成文本（completion）的最大 token 数
            per_device_train_batch_size = per_device_train_batch_size,
            num_generations = num_generations,
            num_train_epochs=num_train_epochs,
            fp16=fp16,
            use_vllm=use_vllm,
            save_strategy=save_strategy,        # 每轮保存一次模型
            eval_strategy=eval_strategy,        # 在每轮结束后跑一次评估
            logging_strategy=logging_strategy,     # 每轮训练结束时记录训练指标
            report_to=report_to,    # 开启 TensorBoard 记录
            logging_dir=f'./data_record/{self.task_name}'# TensorBoard 日志目录
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['dev'],
            processing_class=self.tokenizer,
            callbacks=[CSVLoggerCallback(
                train_csv_path=f"./data_record/{self.task_name}/train_record.csv",
                eval_csv_path =f"./data_record/{self.task_name}/eval_record.csv"
            )],
        )

        trainer.train()


