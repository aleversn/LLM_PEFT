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
import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm
from main.loader import AutoDataloader
from main.analysis import Analysis


class Trainer():

    def __init__(self, tokenizer, config, from_pretrained, loader_name, data_path, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1, eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.loader_name = loader_name
        self.model_from_pretrained = from_pretrained
        self.data_path = data_path
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.eval_mode = eval_mode
        self.dataloader_init()
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

    def dataloader_init(self):
        d = AutoDataloader(self.tokenizer, self.config, loader_name=self.loader_name, data_path=self.data_path,
                           max_length=self.max_length)
        self.train_loader, self.eval_loader = d(
            self.batch_size, self.batch_size_eval, self.eval_mode)

    def model_to_device(self, gpu=[0]):
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu)
        self.model.cuda()

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
            eval_scores = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for it in train_iter:
                pad_len = (it['input_ids'] == 0).sum(1).min()
                kep_len = it['input_ids'].size(1) - pad_len
                for key in it.keys():
                    it[key] = it[key][:, :kep_len]
                    it[key] = it[key].to(self.model.module.device)
                
                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(logits, labels)
                for k, v in metrics.items():
                    eval_scores[k] += v

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, **{k: v / train_count for k, v in eval_scores.items()})

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
                **{k: v / train_count for k, v in eval_scores.items()}
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
            eval_scores = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            for it in eval_iter:
                pad_len = (it['input_ids'] == 0).sum(1).min()
                kep_len = it['input_ids'].size(1) - pad_len
                for key in it.keys():
                    it[key] = it[key][:, :kep_len]
                    it[key] = self.cuda(it[key])

                output = self.model(**it)
                loss = output.loss
                loss = loss.mean()

                logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(logits, labels)
                for k, v in metrics.items():
                    eval_scores[k] += v

                eval_loss += loss.data.item()
                eval_count += 1

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, **{k: v / eval_count for k, v in eval_scores.items()})

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                **{k: v / eval_count for k, v in eval_scores.items()}
            })
    
    def compute_metrics(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        pred_logits = shift_logits.argmax(-1)
        pred_logits = pred_logits.tolist()
        shift_labels = labels[..., 1:].tolist()

        metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
        for pred_ids, label_ids in zip(pred_logits, shift_labels):
            try:
                answer_idx = 0
                pad_idx = len(label_ids)
                for i in range(len(label_ids)):
                    if label_ids[i] != -100:
                        answer_idx = i if answer_idx == 0 else answer_idx
                        pad_idx = i + 1 if i + 1 < len(label_ids) else i
                    else:
                        if answer_idx != 0:
                            break                        
                pred_ids = pred_ids[answer_idx:pad_idx]
                label_ids = label_ids[answer_idx:pad_idx]
                pred_txt = self.tokenizer.decode(pred_ids).strip()
                label_txt = self.tokenizer.decode(label_ids).strip()
                pred_tokens = list(jieba.cut(pred_txt))
                label_tokens = list(jieba.cut(label_txt))
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
                for k, v in scores[0].items():
                    metrics_dct[k].append(round(v['f'] * 100, 4))
                metrics_dct['bleu-4'].append(
                    sentence_bleu(
                        [label_tokens],
                        pred_tokens,
                        smoothing_function=SmoothingFunction().method3,
                    )
                )
            except:
                continue
        return {k: np.mean(v) if len(v) > 0 else 0 for k, v in metrics_dct.items()}

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
