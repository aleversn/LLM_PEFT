import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List


class Predictor():

    def __init__(self,
                 num_gpus: list = [0],
                 model_from_pretrained: str = None,
                 **args
                 ):
        '''
        Predictor: ChatGLM预测器 (ChatGLM predictor)

        ### Args:

        `num_gpus`: 使用的GPU编号列表 (the list of GPU numbers)

        `model_config_file_name`: bert配置文件名 (bert config file name)
        '''
        self.num_gpus = num_gpus
        self.model_from_pretrained = model_from_pretrained
        self.model_init()

    def model_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True).cuda()
        self.model_to_device(gpu=self.num_gpus)
        self.model = self.model.eval()

    def model_to_device(self, gpu=[0]):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.model.to(self.device)
        self.true_model = self.model.module if hasattr(
            self.model, 'module') else self.model
    
    def process_model_outputs(self, inputs, outputs, tokenizer):
        responses = []
        for input_ids, output_ids in zip(inputs.input_ids, outputs):
            response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
            responses.append(response)
        return responses
    
    def build_chat_input(self, query:str, history=None):
        if history is None:
            history = []
        history.append(query)
        max_input_tokens = 0
        new_batch_input = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        return new_batch_input, max_input_tokens

    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        with torch.no_grad():
            if build_message:
                inputs = []
                batch_max_len = 0
                for i, t in enumerate(query):
                    if isinstance(t, str):
                        t = {'role': 'user', 'content': t}
                    if history is not None and len(history) > 0:
                        h_unit = history[i]
                    else:
                        h_unit = []
                    t, max_input_tokens = self.build_chat_input(t, h_unit)
                    if batch_max_len < max_input_tokens:
                        batch_max_len = max_input_tokens
                    inputs.append(t)
            else:
                inputs = query
                batch_max_len = 0
                for i in range(len(query)):
                    if len(query[i]) > batch_max_len:
                        batch_max_len = len(query[i])
            batched_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=batch_max_len).to(self.device)
            batched_outputs = self.true_model.generate(**batched_inputs, **{
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'top_p': top_p,
                "temperature": temperature,
                "eos_token_id": self.true_model.config.eos_token_id
            })
            batched_response = self.process_model_outputs(batched_inputs, batched_outputs, self.tokenizer)
        return batched_response

    @torch.inference_mode()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        response, history = self.true_model.chat(
            self.tokenizer, query, history, role, max_length, num_beams, do_sample, top_p, temperature, logits_processor, **kwargs)
        return response, history

    @torch.inference_mode()
    def stream_chat(self, query: str, history: List[Tuple[str, str]] = None, role: str = "user",
                    past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        for result in self.true_model.stream_chat(self.tokenizer, query, history, role, past_key_values, max_length, do_sample, top_p, temperature, logits_processor, return_past_key_values, **kwargs):
            yield result

    def __call__(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=False):
        return self.predict(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)
