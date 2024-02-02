import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
from tqdm import tqdm
from CC.loader import AutoDataloader
from CC.analysis import Analysis


class Trainer():

    def __init__(self, tokenizer, config, from_pretrained, loader_name, data_path, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1, eval_mode='dev', task_name='Sim'):
        self.loader_name = loader_name
        self.model_from_pretrained = from_pretrained
        self.data_path = data_path
        self.task_name = task_name
        self.max_length = max_length
        self.dataloader_init(tokenizer, config, loader_name, data_path,
                             max_length, batch_size, batch_size_eval, eval_mode)
        self.model_init(resume_path=resume_path)
        self.analysis = Analysis()

    def model_init(self, resume_path=None):
        print('AutoModel Choose Model: {}\n'.format(self.model_from_pretrained))
        self.model = AutoModel.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True).half().cuda()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(self.model, resume_path, config=peft_config)
        else:
            self.model = get_peft_model(self.model, peft_config)

    def dataloader_init(self, tokenizer, config, loader_name, data_path, max_length, batch_size, batch_size_eval, eval_mode):
        d = AutoDataloader(tokenizer, config, loader_name=loader_name, data_path=data_path,
                           max_length=max_length)
        self.train_loader, self.eval_loader = d(
            batch_size, batch_size_eval, eval_mode)

    def model_to_device(self, gpu=[0]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(device)

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, gpu=[0], eval_call_epoch=None):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, gpu=gpu, eval_call_epoch=eval_call_epoch)

    def train(self, resume_step=None, num_epochs=30, lr=1e-4, gpu=[0], eval_call_epoch=None):
        self.model_to_device(gpu=gpu)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count)

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
            })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        save_path = f'./save_model/{dir}/ChatGLM_{current_step}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model = self.model.module if hasattr(
            self.model, 'module') else self.model
        save_model.save_pretrained(
            save_path)
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, pure_eval=False, gpu=[0]):
        if pure_eval:
            self.model_to_device(gpu=gpu)

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                for key in it.keys():
                    it[key] = self.cuda(it[key])

                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                eval_loss += loss.data.item()
                eval_count += 1

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count)

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count
            })

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
