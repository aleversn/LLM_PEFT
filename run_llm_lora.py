# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main.trainer.llm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig
import datetime

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/llama3.2-3b/", trust_remote_code=True)
config = AutoConfig.from_pretrained("/root/autodl-tmp/models/llama3.2-3b/", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/root/autodl-tmp/models/llama3.2-3b/', loader_name='LLM_Chat', data_path='Wiki_Humans_SFT_100', max_length=1200, batch_size=4, task_name='Wiki-humans-llama-sft-100' + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

for i in trainer(num_epochs=5):
    a = i

# %%
