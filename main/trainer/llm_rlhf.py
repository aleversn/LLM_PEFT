import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import numpy as np
import jieba
import random
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm
from main.loader import AutoDataloader
from main.models.llm_rlhf import CriticModel, RewardModel, PPO
from main.analysis import Analysis
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import os
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
dist.init_process_group(backend='nccl', rank=0, world_size=1)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
setup_seed(3407)


class Trainer():

    def __init__(self, tokenizer, config, from_pretrained, reward_from_pretrained, loader_name, data_path,\
                  ratio_for_rlhf=-1.0, actor_resume_path=None,critic_resume_path=None, critic_layers_keep=1,resume_path=None, max_length=512, batch_size=1, batch_size_eval=1, eval_mode='dev', task_name='Sim'):
        self.tokenizer = tokenizer
        self.config = config
        self.accelerate = accelerator
        self.loader_name = loader_name
        self.data_path = data_path
        self.model_from_pretrained = from_pretrained
        self.reward_from_pretrained = reward_from_pretrained
        self.critic_layers_keep = critic_layers_keep
        self.resume_path = resume_path
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
        # 加入tensorboard监控训练过程
        if not os.path.exists('logs/tensorboard_logs'):
            os.mkdir('logs/tensorboard_logs')
        self.writer = SummaryWriter(os.path.join('logs/tensorboard_logs', task_name))

    # 初始化模型
    def model_init(self, actor_resume_path=None, critic_resume_path=None):
        if self.accelerate.is_local_main_process:
            print('AutoModel Choose Model: {}\n'.format(self.model_from_pretrained))
        # 初始化actor模型
        if self.config.model_type == 'qwen2':
            self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_from_pretrained, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=False, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(
                    self.model_from_pretrained, torch_dtype=torch.bfloat16,trust_remote_code=True)
        # 初始化critic模型
        self.critic_model = CriticModel(model_from_pretrained=self.model_from_pretrained, resume_path=critic_resume_path, layers_keep=self.critic_layers_keep)
        # 初始化reward模型
        self.reward_model = RewardModel(model_from_pretrained=self.reward_from_pretrained)
        self.ppo = PPO(self.tokenizer, self.qa_logs)
        # lora相关参数设置
        if self.config.model_type == 'chatglm':
            lora_module = ['query_key_value']
        elif self.config.model_type == 'qwen2':
            lora_module = ['q_proj', 'k_proj', 'v_proj']
        else:
            raise NotImplementedError(f'{self.config.model_type} series has no default lora settings.')
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            target_modules= lora_module,
            lora_alpha=32,
            lora_dropout=0.1
        )
        # 确定是否使用已有权重继续训练，然后加载相应的训练设置
        if actor_resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(actor_resume_path))
            self.model.enable_input_require_grads()
            self.model = PeftModel.from_pretrained(self.model, actor_resume_path, config=peft_config, is_trainable=True)
            self.model.print_trainable_parameters()
        else:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    # 初始化数据集
    def dataloader_init(self):
        d = AutoDataloader(self.tokenizer, self.config, loader_name=self.loader_name, data_path=self.data_path,
                           max_length=self.max_length)
        self.train_loader, self.eval_loader = d(
            self.batch_size, self.batch_size_eval, self.eval_mode, True)

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, eval_call_epoch=None, weight_for_cos_and_jaccard=[0.5, 0.5], ppo_epochs=3, ppo_epsilon=0.15, alpha=0.5, beta=0.5, gamma=0):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_epoch=eval_call_epoch, ppo_epochs=ppo_epochs, weight_for_cos_and_jaccard=weight_for_cos_and_jaccard, ppo_epsilon=ppo_epsilon, alpha=alpha, beta=beta, gamma=gamma)

    # 对模型进行训练
    def train(self, resume_step=None, num_epochs=30, lr=1e-4, num_beams=1, num_return_sequences=1,eval_call_epoch=None, weight_for_cos_and_jaccard=[0.5, 0.5], ppo_epochs=3, ppo_epsilon=0.15, alpha=0.5, beta=0.5, gamma=0):
        # 设置优化器，指定要训练的参数
        # 在本工作中，RL仅优化actor模型、critic模型的输出层
        optimizer = optim.Adam(list(self.model.parameters()) + list(self.critic_model.output_linear.parameters()), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        # 指定参与分布式训练的组件
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
            # 对需要训练的模型开启train模式
            self.model.train()
            self.critic_model.train()

            for it in train_iter:
                is_rlhf = torch.tensor([random.randint(0, 10) / 10]).to(self.accelerate.device)
                self.accelerate.wait_for_everyone()
                dist.broadcast(is_rlhf, src=0)
                is_rlhf = is_rlhf.item() <= self.ratio_for_rlhf
                for loss, logits in self.ppo(is_rlhf, self.model, self.reward_model, self.critic_model, **it, num_beams=num_beams, num_return_sequences=num_return_sequences, weight_for_cos_and_jaccard=weight_for_cos_and_jaccard, ppo_epochs=ppo_epochs, epsilon=ppo_epsilon, alpha=alpha, beta=beta, gamma=gamma):
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

                self.writer.add_scalar('train/per_step/loss', train_loss / train_count, global_step=train_step)
                self.writer.add_scalars('train/per_step/metric', {k: v / train_count for k, v in eval_scores.items()}, global_step=train_step)

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count, **{k: v / train_count for k, v in eval_scores.items()})
            self.writer.add_scalar('train/per_epoch/loss', train_loss / train_count, global_step=epoch + 1)
            self.writer.add_scalars('train/per_epoch/metric', {k: v / train_count for k, v in eval_scores.items()}, global_step=epoch + 1)
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
            print('work dir:', os.getcwd())
            # 存在路径问题，如果运行文件位置发生变化，这里的路径也要做出适当修改
            qa_logs_path = f'data_record/{current_uid if self.task_name is None else self.task_name}'
            if not os.path.exists(qa_logs_path):
                os.makedirs(qa_logs_path)
            with open(os.path.join(qa_logs_path, f'{train_step}_log'), encoding='utf-8', mode='w+') as f:
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
        save_path = f'save_model/{dir}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        peft_save_path = os.path.join(save_path, f'Actor_{current_step}')
        if not os.path.exists(peft_save_path):
            os.makedirs(peft_save_path)
        actor_model = self.accelerate.unwrap_model(self.model)
        actor_model.save_pretrained(
            peft_save_path,
            is_main_process=self.accelerate.is_main_process,
            save_function=self.accelerate.save,
        )
        critic_save_path = os.path.join(save_path, f'Critic_{current_step}')
        if not os.path.exists(critic_save_path):
            os.makedirs(critic_save_path)
        critic_model = self.accelerate.unwrap_model(self.critic_model)
        critic_linear = critic_model.output_linear
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
            self.writer.add_scalar('eval/per_epoch/loss', eval_loss / eval_count, global_step=epoch + 1)
            self.writer.add_scalars('eval/per_epoch/metric', {k: v / eval_count for k, v in eval_scores.items()}, global_step=epoch + 1)
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

        try:
            metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
            for pred_ids, label_ids in zip(pred_logits, shift_labels):
                pred_ids_non_mask = []
                label_ids_non_mask = []
                for i in range(len(label_ids)):
                    if label_ids[i] != -100:
                        pred_ids_non_mask.append(pred_ids[i])
                        label_ids_non_mask.append(label_ids[i])
                pred_txt = self.tokenizer.decode(pred_ids_non_mask).strip()
                label_txt = self.tokenizer.decode(label_ids_non_mask).strip()
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
                    ) * 100
                )
            return {k: np.mean(v) for k, v in metrics_dct.items()}
        except:
            return {k: 0 for k, v in metrics_dct.items()}