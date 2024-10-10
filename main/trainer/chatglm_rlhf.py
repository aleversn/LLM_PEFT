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
import random
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm
from main.loader import AutoDataloader
from main.models.chatglm_rlhf import CriticModel, RewardModel, PPO
from main.analysis import Analysis
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])


class Trainer():

    def __init__(self, tokenizer, config, from_pretrained, reward_from_pretrained, loader_name, data_path, ratio_for_rlhf=0.4, actor_resume_path=None, critic_resume_path=None, critic_layers_keep=1, max_length=512, batch_size=1, batch_size_eval=1, eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.loader_name = loader_name
        self.data_path = data_path
        self.model_from_pretrained = from_pretrained
        self.reward_from_pretrained = reward_from_pretrained
        self.critic_layers_keep = critic_layers_keep
        self.data_path = data_path
        self.ratio_for_rlhf = ratio_for_rlhf
        self.task_name = task_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.eval_mode = eval_mode
        self.decay_up_matrix_T = None
        self.dataloader_init()
        self.qa_logs = {}
        self.model_init(actor_resume_path=actor_resume_path, critic_resume_path=critic_resume_path)
        self.analysis = Analysis()

    def model_init(self, actor_resume_path=None, critic_resume_path=None):
        if self.accelerate.is_local_main_process:
            print('AutoModel Choose Model: {}\n'.format(self.model_from_pretrained))
        self.model = AutoModel.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        self.critic_model = CriticModel(model_from_pretrained=self.model_from_pretrained, resume_path=critic_resume_path, layers_keep=self.critic_layers_keep)
        self.reward_model = RewardModel(model_from_pretrained=self.reward_from_pretrained)
        self.ppo = PPO(self.tokenizer, self.qa_logs)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            target_modules=['query_key_value'],
            lora_alpha=32,
            lora_dropout=0.1
        )
        if actor_resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(actor_resume_path))
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(self.model, actor_resume_path, config=peft_config)
        else:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def dataloader_init(self):
        d = AutoDataloader(self.tokenizer, self.config, loader_name=self.loader_name, data_path=self.data_path,
                           max_length=self.max_length)
        self.train_loader, self.eval_loader = d(
            self.batch_size, self.batch_size_eval, self.eval_mode, True)
        
    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, eval_call_epoch=None, ppo_epochs=5):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_epoch=eval_call_epoch, ppo_epochs=ppo_epochs)

    def train(self, resume_step=None, num_epochs=30, lr=1e-4, num_beams=3, num_return_sequences=2, eval_call_epoch=None, ppo_epochs=5):

        optimizer = optim.Adam(list(self.model.parameters()) + list(self.critic_model.output_linear.parameters()), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        self.model, self.critic_model, self.reward_model, optimizer, train_loader, scheduler, self.ppo = self.accelerate.prepare(self.model, self.critic_model, self.reward_model, optimizer, self.train_loader, scheduler, self.ppo)

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

            train_iter = tqdm(train_loader)
            self.model.train()

            for it in train_iter:
                is_rlhf = random.randint(0, 10) / 10 <= 0.4
                if is_rlhf:
                    it['input_ids'] = it['input_ids_without_last_turn']
                    max_new_tokens = torch.max(it['last_input_len']).item()
                    sequences, log_probs, gen_texts, logits = self.ppo.generate_with_rlhf(self.model, it["input_ids"], it["query"], num_beams=num_beams, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens)
                
                else:
                    max_new_tokens = torch.max(it['last_input_len']).item()
                    sequences, log_probs, gen_texts, logits = self.ppo.generate_with_ft(self.model, it["input_ids"], it["last_assistant_content"], max_new_tokens)
                
                # compute reward for generated sequences
                gold_answers = it["gold_answers"]
                bad_answers = it["bad_answers"]
                reward = []
                for i in range(self.batch_size):
                    if is_rlhf:
                        b_gen_texts = gen_texts[i * num_return_sequences : (i + 1) * num_return_sequences]
                    else:
                        b_gen_texts = [gen_texts[i]]
                    b_gold_answers = gold_answers[i]
                    b_bad_answers = bad_answers[i]
                    b_reward = self.reward_model(gen_texts=b_gen_texts, gold_answers=b_gold_answers, bad_answers=b_bad_answers).unsqueeze(1)
                    reward.append(b_reward)
                reward = torch.cat(reward, dim=0)
                assert reward.shape == (len(gen_texts), 1), "need unsqueeze for next scatter_"
                rewards = torch.zeros_like(sequences, dtype=reward.dtype)
                pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
                masks = (sequences!=pad_id).long()
                final_position = (sequences[:,it['input_ids'].size(-1):] != pad_id).sum(dim=-1) + it['input_ids'].size(-1) - 1
                index = final_position.unsqueeze(-1)
                rewards.scatter_(dim=1, index=index, src=reward)
                # 确保都放到values所在的device

                torch.cuda.empty_cache()
                
                for ppo_epoch in range(ppo_epochs):
                    # compute new log probs
                    new_log_probs, _ = self.ppo.get_log_probs_with_input_ids(self.model, sequences, log_probs.shape[1])
                    entropy = 0 # 暂时不需要熵的约束
                    # compute value
                    # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
                    # 生成序列同时包括state和next action
                    # prepare input for critic model
                    input_ids_critic = sequences
                    values = self.critic_model(input_ids=input_ids_critic)
                    # compute gae
                    gae = self.ppo.gae_vectorize(values=values, rewards=rewards, masks=masks)
                    advantages = gae[:, -log_probs.shape[-1]:]
                    # 计算value的估计量的偏差作为actor loss
                    # 以及ppo的actor_loss
                    value_estimator_delta = advantages
                    ratio = (new_log_probs - log_probs).exp()
                    # print("reward",reward, "ratio:", ratio, sep="\n")
                    if torch.isinf(ratio).any():
                        break
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                    actor_loss  = - torch.min(surr1, surr2).mean()
                    critic_loss = value_estimator_delta.square().mean()
                    loss = 0.5 * (critic_loss + actor_loss) - 0.001 * entropy
                    # optimize
                    optimizer.zero_grad()
                    # loss.backward()
                    self.accelerate.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    self.critic_model.zero_grad()
                
                if not is_rlhf:
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
            import json
            with open(f'./data_record/{current_uid if self.task_name is None else self.task_name}/{train_step}_log', encoding='utf-8', mode='w+') as f:
                for item in self.qa_logs:
                    f.write(json.dumps({f'{item}': self.qa_logs[item]}, ensure_ascii=False) + '\n')
            self.qa_logs.clear()
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    @accelerator.on_local_main_process
    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        save_dir = f'./save_model/{dir}'
        peft_save_path = os.path.join(save_dir, f'ChatGLM_{current_step}')
        if not os.path.exists(peft_save_path):
            os.makedirs(peft_save_path)
        save_model = self.accelerate.unwrap_model(self.model)
        save_model.save_pretrained(
            peft_save_path,
            is_main_process=self.accelerate.is_main_process,
            save_function=self.accelerate.save,
        )
        critic_save_path = os.path.join(save_dir, f'Critic_{current_step}')
        if not os.path.exists(critic_save_path):
            os.makedirs(critic_save_path)
        critic_linear = self.critic_model.output_linear
        torch.save(critic_linear.state_dict(), os.path.join(critic_save_path, 'linear.pth'))
        self.analysis.append_model_record(current_step)
        return current_step   

    def eval(self, epoch, pure_eval=False):
        if pure_eval:
            self.model = self.accelerate.prepare_model(self.model)
        self.eval_loader = self.accelerate.prepare_data_loader(self.eval_loader)

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

                output = self.model(input_ids=it['input_ids'], labels=it['labels'])
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
                for i in range(len(label_ids)):
                    if label_ids[i] != -100:
                        answer_idx = i
                        break
                pred_ids = pred_ids[answer_idx:]
                label_ids = label_ids[answer_idx:]
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