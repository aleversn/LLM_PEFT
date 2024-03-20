from CC.trainer.chatglm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='./model/chatglm3-6b', resume_path='./save_model/FDRAG/ChatGLM_25746', loader_name='ChatGLM_Chat', data_path='FDRAG', max_length=3600, batch_size=1, task_name='FDRAG')

for i in trainer(num_epochs=60, resume_step=25746, lr=1e-5):
    a = i