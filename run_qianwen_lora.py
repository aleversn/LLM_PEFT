from CC.trainer.qianwen_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/Qwen-14B-Chat-Int4", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/Qwen-14B-Chat-Int4", trust_remote_code=True)
config.disable_exllama = True
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='./model/Qwen-14B-Chat-Int4', loader_name='Qianwen_Chat', data_path='FD', max_length=512, batch_size=1, task_name='FD_Qianwen')

for i in trainer(num_epochs=30):
    a = i