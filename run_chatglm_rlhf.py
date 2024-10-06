# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from main.trainer.chatglm_rlhf import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/models/chatglm3-6b/', reward_from_pretrained='/home/lpc/models/text2vec-base-chinese/', loader_name='ChatGLM_RLHF', data_path='ID', max_length=1200, batch_size=2, task_name='ID')

for i in trainer(num_epochs=5):
    a = i

# %%
