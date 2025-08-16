import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
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
        self.model_init(**args)

    def model_init(self, **args):
        self.config = AutoConfig.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_from_pretrained, padding_side="left", trust_remote_code=True)
        self.model_type = self.config.model_type if hasattr(self.config, 'model_type') else 'llama'
        self.llm = LLM(self.model_from_pretrained, tensor_parallel_size=self.tp, trust_remote_code=True, enable_lora=self.lora_path is not None, **args)
        self.processor = self.tokenizer
        if hasattr(self.config, 'eos_token_id'):
            self.eos_token_id = [self.config.eos_token_id]
        if hasattr(self.config, 'bos_token_id'):
            self.bos_token_id = [self.config.bos_token_id]
        if self.model_type == 'chatglm':
            self.eos_token_id = self.config.eos_token_id
        elif self.model_type == 'llama':
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
        elif self.model_type == "mimo":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
        elif self.model_type == "tinyr1":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
        elif self.model_type == "qwen2_5_vl":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            self.processor = AutoProcessor.from_pretrained(self.model_from_pretrained, padding_side='left')
        elif self.model_type == "llava":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            from transformers import LlavaForConditionalGeneration
            self.model =  LlavaForConditionalGeneration.from_pretrained(self.model_from_pretrained, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_from_pretrained, padding_side='left')
        elif self.model_type == "mllama":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            from transformers import MllamaForConditionalGeneration
            self.model =  MllamaForConditionalGeneration.from_pretrained(self.model_from_pretrained, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_from_pretrained, padding_side='left', trust_remote_code=True)
    
    def process_model_outputs(self, inputs, outputs, tokenizer):
        responses = []
        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
            responses.append(response)
        return responses
        
    def build_chat_input(self, query:list, history=None):
        if history is None:
            history = []
        history.append(query)
        is_mm = self.is_mm(history)
        max_input_tokens = 0
        new_batch_input = self.tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        return {
            'text': new_batch_input,
            'max_input_tokens': max_input_tokens,
            'is_mm': is_mm
        }
    
    def is_mm(self, message: dict | list):
        if not isinstance(message, list):
            message = [message]
        for msg in message:
            if isinstance(msg, dict) and 'content' in msg and isinstance(msg['content'], list):
                for content in msg['content']:
                    if 'type' in content and content['type'] in ['image', 'video']:
                        return True
        return False
    
    def process_mm(self, query_list:list, history_list=None):
        if history_list is not None:
            assert len(query_list) == len(history_list), "Query and history must have the same length."
            for query, history in zip(query_list, history_list):
                if isinstance(query, str):
                    query = {'role': 'user', 'content': query}
                history.append(query)
        else:
            history_list = [[query] for query in query_list]
        
        images_list = []
        for history in history_list:
            images = []
            for msg in history:
                if isinstance(msg['content'], list):
                    for content in msg['content']:
                        if isinstance(content, dict) and 'type' in content and content['type'] == 'image':
                            images.append(convert_image_mode(Image.open(content['image']), "RGB"))
            images_list.append(images)
        return images_list

    def prepare_generate(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 1.0, temperature=0, do_sample: bool = False, build_message=True):
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        is_mm = False
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
                chat_input = self.build_chat_input(t, h_unit)
                if batch_max_len < chat_input['max_input_tokens']:
                    batch_max_len = chat_input['max_input_tokens']
                inputs.append(chat_input['text'])
                if chat_input['is_mm']:
                    is_mm = True
        else:
            inputs = query
            batch_max_len = 0
            for i in range(len(query)):
                if len(query[i]) > batch_max_len:
                    batch_max_len = len(query[i])
        
        if is_mm:
            images_list = self.process_mm(query, history)
        else:
            images_list = []
        
        if len(images_list) > 0:
            new_inputs = []
            for i, text in enumerate(inputs):
                new_inputs.append({
                    "prompt": text,
                    "multi_modal_data": {"image": images_list[i]},
                })
            inputs = new_inputs
        
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

        return inputs, sampling_params
    
    def predict(self, query: str | list = '', history: List = None, max_length=512, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True):
        '''
        Predict method for generating responses from the model.

        ### Args:
        - `query`: The input query or list of queries.
        - `history`: Conversation history, if any.
        - `max_length`: Maximum length of the generated response.
        - `max_new_tokens`: Maximum number of new tokens to generate.
        - `num_beams`: Number of beams for beam search (default is 1).
        - `top_p`: Top-p sampling parameter.
        - `temperature`: Temperature for sampling.
        - `do_sample`: Whether to use sampling or greedy decoding.
        - `build_message`: Whether to build the message format for the model.

        ### Returns:
        - List of generated responses.
        '''

        inputs, sampling_params = self.prepare_generate(query, history, max_length, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)
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
