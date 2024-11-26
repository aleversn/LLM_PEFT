# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main.trainer.llm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/Meta-Llama-3-8B-Instruct/", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/lpc/models/Meta-Llama-3-8B-Instruct/", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/models/Meta-Llama-3-8B-Instruct/', loader_name='LLM_Chat', data_path='FDEX2', max_length=3600, batch_size=4, task_name='FDEX2')

for i in trainer(num_epochs=5):
    a = i

# %%
