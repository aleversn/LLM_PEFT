import json
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Tuple, List


class Predictor():

    def __init__(self,
                 tensor_parallel_size: int = 1,
                 model_from_pretrained: str = None,
                 resume_path: str = None,
                 **args
                 ):
        '''
        Predictor: LLM预测器 (LLM predictor)

        ### Args:

        `tensor_parallel_size`: 张量并行使用的 GPU 数量（模型被拆分到多个卡）

        `model_from_pretrained`: 预训练模型的路径或名称（如HuggingFace模型名）

        `resume_path`: 如果有预训练的LoRA模型，可以指定路径进行加载
        '''
        self.tp = tensor_parallel_size
        self.model_from_pretrained = model_from_pretrained
        self.lora_path = resume_path
        self.model_init()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_from_pretrained, padding_side="left", trust_remote_code=True)
        self.llm = LLM(self.model_from_pretrained, tensor_parallel_size=self.tp, trust_remote_code=True, enable_lora=self.lora_path is not None)
        if hasattr(self.config, 'eos_token_id'):
            self.eos_token_id = [self.config.eos_token_id]
        if hasattr(self.config, 'bos_token_id'):
            self.bos_token_id = [self.config.bos_token_id]
        if self.config.model_type == 'chatglm':
            self.eos_token_id = self.config.eos_token_id
        elif self.config.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if not hasattr(self, 'eos_token_id'):
                self.eos_token_id = []
            for t in terminators:
                if t is not None:
                    self.eos_token_id.append(t)
        elif self.config.model_type == "mimo":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
        elif self.config.model_type == "tinyr1":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
    
    def process_model_outputs(self, inputs, outputs, tokenizer):
        responses = []
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
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

    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 1.0, temperature=0, do_sample: bool = False, build_message=True):
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
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

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

        if self.lora_path is not None:
            outputs = self.llm.generate(
                inputs,
                sampling_params,
                lora_request=LoRARequest("custom_lora", 1, self.lora_path)
            )
        else:
            outputs = self.llm.generate(inputs, sampling_params)
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append(generated_text)
        return results

    def __call__(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True):
        return self.predict(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)
