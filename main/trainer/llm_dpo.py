import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM
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
from main.models.DPOLoss import DPOLoss 
from accelerate import cpu_offload_with_hook
accelerator = Accelerator()


class Trainer():

    def __init__(self, tokenizer, from_pretrained, loader_name, data_path, config=None, resume_path=None, max_length=512, batch_size=1, batch_size_eval=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.loader_name = loader_name
        self.data_path = data_path
        self.from_pretrained = from_pretrained
        self.data_path = data_path
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
            base = self.model
            self.policy_model = PeftModel.from_pretrained(
                base, resume_path, config=peft_config, is_trainable=True)
            self.policy_model.enable_input_require_grads()
            base_copy = copy.deepcopy(base)
            self.reference_model = PeftModel.from_pretrained(
                base_copy, resume_path, config=peft_config)
            for p in self.reference_model.parameters():
                p.requires_grad = False
            self.reference_model.eval()
            self.policy_model.print_trainable_parameters()
            self.reference_model.print_trainable_parameters()
        else:
            base = self.model
            self.policy_model = get_peft_model(base, peft_config)
            base_copy = copy.deepcopy(base)
            self.reference_model = get_peft_model(base_copy, peft_config)
            for p in self.reference_model.parameters():
                p.requires_grad = False
            self.reference_model.eval()
            self.policy_model.print_trainable_parameters()
            self.reference_model.print_trainable_parameters()
        
    def dataloader_init(self):
        d = AutoDataloader(self.tokenizer, self.config, loader_name=self.loader_name, data_path=self.data_path,
                           max_length=self.max_length)
        self.train_loader, self.eval_loader = d(
            self.batch_size, self.batch_size_eval, self.eval_mode, True)

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, eval_call_epoch=None, beta=0.1):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_epoch=eval_call_epoch, beta=beta)
    
    def compute_logprobs(self, logits, labels, mask=None):
        """
        logits:  shape (batch_size, sequence_len, vocab_size)，即将label输入给模型后输出的结果
        labels:  shape (batch_size, sequence_len)
        """

        # 需要先进行位移操作
        # 去掉标签的第一个
        labels = labels[:, 1:].clone()
        # 去掉模型输出的最后一个
        logits = logits[:, :-1, :]

        logps = F.log_softmax(logits, dim=-1)

        select_logprobs = torch.gather(
            input=logps,
            dim=-1,
            index=labels.unsqueeze(1)
        ).squeeze(1)

        if mask is not None:
            mask = mask[:, 1:].clone()
            # 进行掩码padding部分
            select_logprobs = select_logprobs * mask
            # 计算和
            average_logprobs = select_logprobs.sum(-1) 
            # 计算平均
            #average_logprobs = select_logprobs.sum(-1) / mask.sum(-1)
            return average_logprobs
        else:
            return select_logprobs.mean(-1)
    
    def compute_batch_loss(self, chosen, rejected, chosen_mask, rejected_mask, policy_model, reference_model, beta):
        policy_chosen_logits  = policy_model(input_ids=chosen).logits
        policy_rejected_logits = policy_model(input_ids=rejected).logits

        # —— Reference 前向（关闭梯度，节省内存）
        with torch.no_grad():
            reference_chosen_logits  = reference_model(input_ids=chosen).logits
            reference_rejected_logits = reference_model(input_ids=rejected).logits

        # logits只是原始分数，这里得到对数概率（先进行softmax,再进行对数）
        policy_chosen_lp  = self.compute_logprobs(policy_chosen_logits,  chosen, chosen_mask)
        policy_rejected_lp = self.compute_logprobs(policy_rejected_logits, rejected, rejected_mask)
        reference_chosen_lp  = self.compute_logprobs(reference_chosen_logits,  chosen, chosen_mask)
        reference_rejected_lp = self.compute_logprobs(reference_rejected_logits, rejected, rejected_mask)

        # DPO 损失
        dpo_loss, chosenr, rrjectedr, policyr, referencer, margin = DPOLoss(beta)(
            policy_chosen_logps=policy_chosen_lp,
            policy_rejected_logps=policy_rejected_lp,
            reference_chosen_logps=reference_chosen_lp,
            reference_rejected_logps=reference_rejected_lp
        )

        return dpo_loss, chosenr, rrjectedr, policyr, referencer, margin


    def train(self, resume_step=None, num_epochs=30, lr=1e-4,  eval_call_epoch=None, beta=0.1):
        '''accumulation_steps = 4  # 每4小批次累积一次梯度
        num_update_steps_per_epoch = len(self.train_loader) // accumulation_steps
        total_train_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = int(0.03 * total_train_steps)  # 比如前 3% steps 用来 warmup'''

        # 优化器只对policy_model参数进行优化
        optimizer = optim.Adam(self.policy_model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        #scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)
        # 2) 激活检查点
        self.policy_model.gradient_checkpointing_enable()
        self.reference_model.gradient_checkpointing_enable()
        # 放到GPU上
        # deepspeed要求只支持 将 单个 模型包装到 ZeRO 优化里
        self.policy_model, optimizer, train_loader, self.eval_loader, scheduler = self.accelerate.prepare(
            self.policy_model, optimizer, self.train_loader, self.eval_loader, scheduler)
        # reference_model 明确放到 GPU 上，但不通过 accelerate.prepare 处理
        self.reference_model = self.reference_model.to(self.accelerate.device)

        self.policy_model.print_trainable_parameters()
        self.reference_model.print_trainable_parameters()
        
        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0

        for epoch in range(num_epochs):
            # 一个epoch的记录
            train_count = 0
            train_loss = 0.0
            # 新增：偏好分类（Preference）正确计数   (总计数为train_count)
            eval_pref_correct = 0
            eval_scores = {
                'chosen_rewards': 0.0,
                'rejected_rewards': 0.0,
                'policy_rewards': 0.0,
                'reference_rewards': 0.0,
                'margin': 0.0
            }
            eval_scores2 = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }
    
            train_iter = tqdm(train_loader)
            # 只对policy_model进行训练
            self.policy_model.train()

            # 当到第26个epoch时，同步 reference_model
            '''if epoch + 1 == 26:
                # 1) 解包：拿到未 wrap（未加 DDP）的 PeftModelForCausalLM
                unwrapped_pol = self.accelerate.unwrap_model(self.policy_model)

                new_ref = copy.deepcopy(unwrapped_pol)
                # 2）把它替换掉旧的 reference_model
                self.reference_model = new_ref
                # 3）冻结权重 & 切到 eval() 模式
                for p in self.reference_model.parameters():
                    p.requires_grad = False
                self.reference_model.eval()
                self.reference_model.print_trainable_parameters()'''

            for idx, it in enumerate(train_iter):
                
                # 一个batch的平均
                dpo_loss, chosen_rewards, rejected_rewards, policy_rewards, reference_rewards, margin = self.compute_batch_loss(
                    chosen=it["chosen"],
                    rejected=it["rejected"],
                    chosen_mask=it["chosen_mask"],
                    rejected_mask=it["rejected_mask"],
                    policy_model=self.policy_model,
                    reference_model=self.reference_model,
                    beta=beta
                )

                # 原始的损失
                output  = self.policy_model(input_ids=it["input_ids"], labels=it["labels"])
                policy_input_logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(policy_input_logits, labels)
                llm_loss = output.loss
                llm_loss = llm_loss.mean()

                loss = 0.4 * llm_loss + 0.6 * dpo_loss
                #loss = dpo_loss
                # loss.backward()
                # 缩放 loss 并累积梯度 :contentReference[oaicite:5]{index=5}
                #loss = loss / accumulation_steps
                self.accelerate.backward(loss)

                '''if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(self.train_loader):
                    optimizer.step()           # 更新权重 :contentReference[oaicite:6]{index=6}
                    scheduler.step()           # 学习率调度
                    #self.policy_model.zero_grad()  # 清空累积的梯度
                    optimizer.zero_grad()'''
                    
                optimizer.step()           # 更新权重 :contentReference[oaicite:6]{index=6}
                scheduler.step()           # 学习率调度
                self.policy_model.zero_grad()  # 清空累积的梯度
                #optimizer.zero_grad()

                #train_loss += loss.data.item() * accumulation_steps
                train_loss += loss.data.item()
                train_count += 1  # train_count是一个epoch中有多少个batch
                train_step += 1   # train_step是总的epoch的batch个数
                
                eval_scores['chosen_rewards']+=chosen_rewards
                eval_scores['rejected_rewards']+=rejected_rewards
                eval_scores['policy_rewards']+=policy_rewards
                eval_scores['reference_rewards']+=reference_rewards
                eval_scores['margin']+=margin

                for k, v in metrics.items():
                    eval_scores2[k] += v
                
                # chosen_rewards和rejected_rewards是每个 batch的平均值，
                # 那么就近似认为这一批整体是“正确”还是“错误”。
                # 这里我们把它当成一个样本：若 chosen_rewards > rejected_rewards 则算作 1， 否则 0。
                correct = 1 if chosen_rewards > rejected_rewards else 0
                eval_pref_correct += correct

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, pref_acc=eval_pref_correct / train_count, **{k: v / train_count for k, v in eval_scores.items()}, **{k: v / train_count for k, v in eval_scores2.items()})

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
                'pref_acc': eval_pref_correct / train_count,
                **{k: v / train_count for k, v in eval_scores.items()},
                **{k: v / train_count for k, v in eval_scores2.items()}
            })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch, beta)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    @accelerator.on_local_main_process
    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        save_path = f'./save_model/{dir}/ChatGLM_{current_step}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存policy_model
        save_model = self.accelerate.unwrap_model(self.policy_model)
        save_model.save_pretrained(
            save_path,
            is_main_process=self.accelerate.is_main_process,
            save_function=self.accelerate.save,
        )
        self.analysis.append_model_record(current_step)
        return current_step

    def eval(self, epoch, pure_eval=False, beta=0.1):
        if pure_eval:
            self.policy_model = self.accelerate.prepare_model(self.policy_model)
        '''self.eval_loader = self.accelerate.prepare_data_loader(
            self.eval_loader)'''

        with torch.no_grad():
            eval_count = 0
            eval_loss = 0
            # 新增：偏好分类（Preference）正确计数   (总计数为eval_count)
            eval_pref_correct = 0
            eval_scores = {
                'chosen_rewards': 0.0,
                'rejected_rewards': 0.0,
                'policy_rewards': 0.0,
                'reference_rewards': 0.0,
                'margin': 0.0
            }
            eval_scores2 = {
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0,
                'bleu-4': 0
            }
    
            eval_iter = tqdm(self.eval_loader)
            self.policy_model.eval()

            for it in eval_iter:

                dpo_loss, chosen_rewards, rejected_rewards, policy_rewards, reference_rewards, margin = self.compute_batch_loss(
                    chosen=it["chosen"],
                    rejected=it["rejected"],
                    chosen_mask=it["chosen_mask"],
                    rejected_mask=it["rejected_mask"],
                    policy_model=self.policy_model,
                    reference_model=self.reference_model,
                    beta=beta
                )
                # 原始的损失
                output  = self.policy_model(input_ids=it["input_ids"], labels=it["labels"])
                policy_input_logits = output.logits
                labels = it['labels']
                metrics = self.compute_metrics(policy_input_logits, labels)
                llm_loss = output.loss
                llm_loss = llm_loss.mean()

                loss = 0.4 * llm_loss + 0.6 * dpo_loss
                #loss = dpo_loss

                eval_loss += loss.data.item()
                eval_count += 1
                
                eval_scores['chosen_rewards']+=chosen_rewards
                eval_scores['rejected_rewards']+=rejected_rewards
                eval_scores['policy_rewards']+=policy_rewards
                eval_scores['reference_rewards']+=reference_rewards
                eval_scores['margin']+=margin

                for k, v in metrics.items():
                    eval_scores2[k] += v
                
                # chosen_rewards和rejected_rewards是每个 batch的平均值，
                # 那么就近似认为这一批整体是“正确”还是“错误”。
                # 这里我们把它当成一个样本：若 chosen_rewards > rejected_rewards 则算作 1， 否则 0。
                correct = 1 if chosen_rewards > rejected_rewards else 0
                eval_pref_correct += correct

                eval_iter.set_description(
                    f'Eval: {epoch + 1}')
                eval_iter.set_postfix(
                    eval_loss=eval_loss / eval_count, pref_acc=eval_pref_correct / eval_count, **{k: v / eval_count for k, v in eval_scores.items()}, **{k: v / eval_count for k, v in eval_scores2.items()})
            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'eval_loss': eval_loss / eval_count,
                'pref_acc': eval_pref_correct / eval_count,
                **{k: v / eval_count for k, v in eval_scores.items()},
                **{k: v / eval_count for k, v in eval_scores2.items()}
            })

    def compute_metrics(self, logits, labels):
        # logits是 bz x seq x vocab
        # 序列的每一个位置都会预测下一个token，每个都会预测vacab个token，值是对应的概率

        # 去掉了每个序列中最后的预测 ，为了跟labels对齐
        shift_logits = logits[..., :-1, :]
        # 得到最大概率的token
        pred_logits = shift_logits.argmax(-1)
        pred_logits = pred_logits.tolist()
        # labels去掉了第一个token，这样就可以和pred_logits对齐
        shift_labels = labels[..., 1:].tolist()

        metrics_dct = {'rouge-1': [], 'rouge-2': [],
                       'rouge-l': [], 'bleu-4': []}
        for pred_ids, label_ids in zip(pred_logits, shift_labels):
            try:
                answer_idx = 0
                # 寻找答案所在位置（也就是assitant的内容）
                for i in range(len(label_ids)):
                    if label_ids[i] != -100:
                        answer_idx = i
                        break
                # 只比对assitant内容，原本的logits是input_ids(包括user内容和assitant内容)然后去预测下一个token概率
                pred_ids = pred_ids[answer_idx:]
                label_ids = label_ids[answer_idx:]
                pred_txt = self.tokenizer.decode(pred_ids).strip()
                label_txt = self.tokenizer.decode(label_ids).strip()
                pred_tokens = list(jieba.cut(pred_txt))
                label_tokens = list(jieba.cut(label_txt))
                rouge = Rouge()
                scores = rouge.get_scores(
                    ' '.join(pred_tokens), ' '.join(label_tokens))
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
