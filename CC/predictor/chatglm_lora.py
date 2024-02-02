from transformers import AutoModel
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
import torch


class Predictor():
    true_model: PeftModelForCausalLM

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 resume_path: str = None
                 ):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_from_pretrained, trust_remote_code=True).half().cuda()
        self.model = PeftModel.from_pretrained(
            self.model, resume_path, config=peft_config)
        self.model_to_device(gpu=num_gpus)

    def model_to_device(self, gpu=[0]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(self.device)
        self.true_model = self.model.module if hasattr(
            self.model, 'module') else self.model

    def predict(self, text='', max_length=150, temperature=1.0):
        with torch.no_grad():
            inputs = self.tokenizer.encode(text)
            input_ids = torch.LongTensor([inputs]).to(self.device)
            output = self.true_model.generate(**{
                'input_ids': input_ids,
                'max_length': max_length,
                'do_sample': False,
                'temperature': temperature
            })
            out_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return out_text

    def __call__(self, text='', max_length=150, temperature=0):
        return self.predict(text=text, max_length=max_length, temperature=temperature)
