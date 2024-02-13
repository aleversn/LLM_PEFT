from CC.trainer.chatglm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='./model/chatglm3-6b', resume_path='./save_model/FD/ChatGLM_74102', loader_name='ChatGLM_Chat', data_path='FD', max_length=3600, batch_size=1, task_name='FD')

for i in trainer(num_epochs=60, lr=1e-5, resume_step=74102):
    a = i