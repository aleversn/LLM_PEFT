# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from main.trainer.llm_rlhf import Trainer
from transformers import AutoTokenizer, AutoConfig
import datetime
import traceback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    tokenizer = AutoTokenizer.from_pretrained(r"/root/autodl-tmp/models/llama3.2-3b-instruct", trust_remote_code=True)
    config = AutoConfig.from_pretrained(r"/root/autodl-tmp/models/llama3.2-3b-instruct", trust_remote_code=True)
    trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained=r"/root/autodl-tmp/models/llama3.2-3b-instruct", \
        reward_from_pretrained=r"/root/autodl-tmp/models/text2vec-base-multilingual", loader_name='LLM_RLHF',\
        data_path='Wiki_Humans_RL_100', ratio_for_rlhf=-1.0, max_length=1024, batch_size=4,  task_name='Wiki-humans-llama3.2-sen-jaccard-100' + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    for i in trainer(num_epochs=50, weight_for_cos_and_jaccard=[0.5, 0.5], ppo_epsilon=0.15, lr=5e-4, ppo_epochs=3, alpha=0.5, beta=0.5, gamma=0): 
        a = i

except Exception as e:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('logs/error_logs'):
        os.mkdir('logs/error_logs')  
    filename = f'logs/error_logs/error_log_{timestamp}.txt'
    with open(filename, 'a') as f:
        exc_info = traceback.format_exc()
        f.write(f'An error occurred: {exc_info}\n')
    print(f'An error occurred. Please check the {filename} file for details.')
# %%
