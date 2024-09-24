from main.trainer.qianwen_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/Qwen-14B-Chat-Int4", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/Qwen-14B-Chat-Int4", trust_remote_code=True)
config.disable_exllama = True
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='./model/Qwen-14B-Chat-Int4', resume_path='./save_model/FDQA_Qianwen/Qwen_3000', loader_name='Qianwen_Chat', data_path='FDQA', max_length=512, batch_size=1, task_name='FDQA_Qianwen')

for i in trainer(lr=3e-4, resume_step=3000, num_epochs=30):
    a = i