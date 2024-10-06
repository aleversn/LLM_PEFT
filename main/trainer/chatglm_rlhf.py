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
from main.models.chatglm_rlhf import CriticModel, RewardModel
from main.analysis import Analysis
from accelerate import Accelerator
accelerator = Accelerator()


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
        self.model_init(actor_resume_path=actor_resume_path, critic_resume_path=critic_resume_path)
        self.analysis = Analysis()
        self.qa_logs = {}

    def model_init(self, actor_resume_path=None, critic_resume_path=None):
        if self.accelerate.is_local_main_process:
            print('AutoModel Choose Model: {}\n'.format(self.model_from_pretrained))
        self.model = AutoModel.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
        self.critic_model = CriticModel(model_from_pretrained=self.model_from_pretrained, resume_path=critic_resume_path, layers_keep=self.critic_layers_keep)
        self.reward_model = RewardModel(model_from_pretrained=self.reward_from_pretrained)
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
        
    def get_log_prob(self, generated_outputs, input_ids, gen_method = "greedy_search"):
        # beam_search generate 给出来的scores就是log_prob了，所以直接gather获取即可
        gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:] 
        # let's stack the logits generated at each step to a tensor
        # 要小心greedy search 拿到的是score，需要再log_softmax
        # 而beam_search 拿到的已经是log_softmax了
        scores = torch.stack(generated_outputs.scores, dim=1)
        # if scores.max() >0 :
        #     gen_method = "greedy_search"
        if gen_method == "beam_search":
            log_prob_stacked = scores
        else:
            log_prob_stacked = torch.stack(generated_outputs.scores, dim=1).log_softmax(dim=-1)
        # now we need to collect the log_prob of the generated token # we need to add a dummy dim in the end to make gather work 
        log_prob = torch.gather(log_prob_stacked, 2, gen_sequences[:, :, None]).squeeze(-1)
        return log_prob
    
    def get_log_probs_with_input_ids(self, states, gen_max_len):
        input_ids = states
        # model_inputs = self.model.prepare_inputs_for_generation(input_ids)
        output = self.model(input_ids)  #将已经生成的序列放进去计算，再次计算得到目标action也就是后续字符的概率或者log_prob值
        logits = output.logits[:, -(gen_max_len+1):-1].log_softmax(dim=-1) # 比先softmax再log好,复杂度减小，并且解决些nan问题
        new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(-1)).squeeze(-1)
        return new_log_probs, output.logits
    
    def process_response(self, output):
        content = ""
        for response in output.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                content = content.replace("[[训练时间]]", "2023年")
            else:
                content = {"name": metadata.strip(), "content": content}
        return content
    
    def generate_with_rlhf(self, input_ids, query, num_beams=1, num_return_sequences=1, max_new_tokens=8):
        '''
        `params:`
            - input_ids: [1, seq_len]
            - query: list, the user query content
            - num_beams: int, 3, 2 # set bigger if you have bigger compute memory
            - num_return_sequences: int, 3, 2 # set bigger if you have bigger compute memory
            - max_new_tokens: int, the max token that LLM can generate for new content
        
        `return:`
            - sequences:  the generated ids of sequences
            - log_probs:  the log_probs of the generated ids
            - gen_texts:  the generated texts of sequences, which clip with max length.
        '''
        assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
        gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
        # 把问题送入模型中，获得问题的输出
        generate_ = self.model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
                            num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                            output_hidden_states=False, return_dict_in_generate=True)
        sequences = generate_.sequences
        log_probs = self.get_log_prob(generated_outputs=generate_, input_ids=input_ids, gen_method=gen_method)
        gen_texts = self.tokenizer.batch_decode(sequences)
        gen_texts = [self.process_response(text) for text in gen_texts]

        for i, q in enumerate(query):
            cur_gen_texts = gen_texts[i * num_return_sequences : (i + 1) * num_return_sequences]
            if q not in self.qa_logs:
                self.qa_logs[q] = []
            self.qa_logs[q] += cur_gen_texts # 将本query的答案保存在qa_logs中；对于同样的query，若多次生成回答，则使用extend方法进行全部存储

        return sequences, log_probs, gen_texts, None

    def generate_with_ft(self, input_ids, last_assistant_content, gen_max_len):
        '''
        the target sentence is directly used to improve the probability of the RL. zh: 目标句直接用RL提升它的概率
        
        `params:`
            - input_ids: [batch_size, seq_len], query ids with answer
            - last_assistant_content: str, the standard answer
            - gen_max_len: str, the max length of answer ids
        
        `return:`
            - sequences:  the generated ids of sequences, in here is the original input_ids
            - log_probs:  the log_probs of the input_ids.
            - gen_texts:  the generated texts of sequences, in here is the original last_assistant_content.
        '''
        sequences = input_ids
        with torch.no_grad():
            log_probs, logits = self.get_log_probs_with_input_ids(input_ids, gen_max_len=gen_max_len)
        should_gen_texts = last_assistant_content
        return sequences, log_probs, should_gen_texts, logits
    
    def get_decay_up_matrix_T(self, dtype=torch.float, device="cpu", max_length=2048, gamma=0.99, tau=0.95):
        ''' 
        生成衰减矩阵
        
        `params:`
            - dtype: torch.dtype
            - device: torch.device
            - max_length: int
            - gamma: float
            - tau: float
        
        `return:`
            - decay_up_matrix_T: torch.Tensor
        '''
        decay = gamma * tau # 衰减系数
        decay_row = torch.ones(max_length, dtype=dtype, device=device)*decay
        decay_row[0] = 1
        decay_row_cross_time = decay_row.cumprod(dim=-1) # 使用cumprod进行连乘，形成(gamma*tau),(gamma*tau)^2,...,(gamma*tau)^2048这样的结构
        assert decay_row_cross_time.sign().min() == 0
        decay_up_matrix = torch.zeros((max_length, max_length), dtype=dtype, device=device)
        for i in range(max_length):
            decay_row = decay_row_cross_time.roll(i)
            decay_row[:i] = 0 # 确保看不见前面的
            decay_up_matrix[i] = decay_row
        decay_up_matrix_T = decay_up_matrix.T # 先进行转置，因为后面需要用到矩阵乘法
        return decay_up_matrix_T
    
    def gae_vectorize(self, values, rewards, masks=None):
        """
        `params:`
            - values: `[batch_size, sequence_length]`, 表示各个时间步状态的状态值。
            - rewards: `[batch_size, sequence_length]`, 表示各个时间步做出的动作的奖励，对于gpt当前动作也是动作对应的下一状态。所以shape和values一样
                    **注意这里的`rewards`表示当前动作状态的`reward`**
            - masks: 由于是要对生成的`actions`做`gae`，也就是泛化优势估计，
                     所以类似以往的`mask`只需要对`padding`进行`mask`，
                     因为`padding`的`delta`会被放入加权计算，而`action`前面的`delta`，
                     由于生成的衰减矩阵就是上三角的，自然就看不到前面的。
                     `0`表示`mask`， `1`表示需要的。
        """
        action_rewards = rewards.roll(-1) # 当前状态的动作的奖励是下一个状态出现时给出的，而奖励是基于状态计算的，所以需要shift一个时间步回去
        # 为了学到最后输出的<eop>,所以给最后的状态赋予一个rewards试试
        action_rewards = (action_rewards + rewards) / 2 # 将奖励分配到最后两步

        values_estimator_1_order = action_rewards + values.roll(-1) # 这里要注意roll是循环的，所以最后一位的值可能不能用
        deltas = values_estimator_1_order - values  #必须要action+下一个时刻的值函数减去当前值函数，这是表示当前action的优势
        # get weight matrix
        if self.decay_up_matrix_T is None:
            self.decay_up_matrix_T = self.get_decay_up_matrix_T(dtype=deltas.dtype, device= deltas.device)
        # 计算gae
        max_goal_length = deltas.shape[-1]
        sub_decay_up_matrix_T = self.decay_up_matrix_T[:max_goal_length, :max_goal_length]
        if masks is not None:
            deltas = deltas * masks
        gae = deltas.matmul(sub_decay_up_matrix_T.to(deltas.device))
        assert gae.shape == deltas.shape
        return gae

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, eval_call_epoch=None, ppo_epochs=5):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, eval_call_epoch=eval_call_epoch, ppo_epochs=ppo_epochs)

    def train(self, resume_step=None, num_epochs=30, lr=1e-4, num_beams=3, num_return_sequences=2, eval_call_epoch=None, ppo_epochs=5):

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)
        self.model, optimizer, train_loader, scheduler = self.accelerate.prepare(self.model, optimizer, self.train_loader, scheduler)

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
                    sequences, log_probs, gen_texts, logits = self.generate_with_rlhf(it["input_ids"], it["query"], num_beams=num_beams, num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens)
                
                else:
                    max_new_tokens = torch.max(it['last_input_len']).item()
                    sequences, log_probs, gen_texts, logits = self.generate_with_ft(it["input_ids"], it["last_assistant_content"], max_new_tokens)
                
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
                rewards = torch.zeros_like(sequences, dtype=reward.dtype, device=reward.device)
                pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
                masks = (sequences!=pad_id).long().to(reward.device)
                final_position = (sequences[:,it['input_ids'].size(-1):] != pad_id).sum(dim=-1) + it['input_ids'].size(-1) - 1
                index = final_position.unsqueeze(-1)
                rewards.scatter_(dim=1, index=index, src=reward)
                # 确保都放到values所在的device
                rewards = torch.tensor(rewards, dtype=self.critic_model.dtype, device=self.critic_model.device)
                masks = masks.to(self.critic_model.device)

                torch.cuda.empty_cache()
                
                for ppo_epoch in range(ppo_epochs):
                    # compute new log probs
                    new_log_probs, _ = self.get_log_probs_with_input_ids(sequences, log_probs.shape[1])
                    entropy = 0 # 暂时不需要熵的约束
                    # compute value
                    # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
                    # 生成序列同时包括state和next action
                    # prepare input for critic model
                    input_ids_critic = sequences.to(self.critic_model.device)
                    values = self.critic_model(input_ids=input_ids_critic)
                    # compute gae
                    gae = self.gae_vectorize(values=values, rewards=rewards, masks=masks)
                    advantages = gae[:, -log_probs.shape[-1]:].to(new_log_probs.device)
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
            self.qa_logs = {}
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