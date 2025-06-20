import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM, TextIteratorStreamer, AutoProcessor
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
import threading
from typing import Tuple, List, Optional

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    print("Qwen2_5_VLForConditionalGeneration is not available. Please install the transformers>=4.47.")

class Predictor():

    def __init__(self,
                 model_from_pretrained: str = None,
                 peft_path: str = None,
                 **args
                 ):
        '''
        Predictor: LLM预测器 (LLM predictor)

        ### Args:

        `model_config_file_name`: bert配置文件名 (bert config file name)
        '''
        self.peft_path = peft_path
        if self.peft_path is not None:
            self.peft_config = LoraConfig.from_pretrained(self.peft_path)
        self.model_from_pretrained = model_from_pretrained
        self.model_init()

    def model_init(self):
        self.config = AutoConfig.from_pretrained(
            self.model_from_pretrained, trust_remote_code=True)
        self.model_type = self.config.model_type if hasattr(self.config, 'model_type') else 'llama'
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_from_pretrained, padding_side="left", trust_remote_code=True)
        self.processor = self.tokenizer
        if hasattr(self.config, 'eos_token_id'):
            self.eos_token_id = [self.config.eos_token_id]
        if hasattr(self.config, 'bos_token_id'):
            self.bos_token_id = [self.config.bos_token_id]
        if self.model_type == 'chatglm':
            self.model = AutoModel.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            self.eos_token_id = self.config.eos_token_id
        elif self.model_type == 'llama':
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_from_pretrained, trust_remote_code=True).to(torch.bfloat16)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if not hasattr(self, 'eos_token_id'):
                self.eos_token_id = []
            for t in terminators:
                if t is not None:
                    self.eos_token_id.append(t)
        elif self.model_type == 'qwen':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.model_type == 'qwen2':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_from_pretrained, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif self.model_type == "mimo":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            self.model =  AutoModelForCausalLM.from_pretrained(self.model_from_pretrained, device_map="auto",trust_remote_code=True)
        elif self.model_type == "tinyr1":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            self.model =  AutoModelForCausalLM.from_pretrained(self.model_from_pretrained, device_map="auto",trust_remote_code=True)
        elif self.model_type == "qwen2_5_vl":
            self.eos_token_id = self.config.eos_token_id
            self.bos_token_id = self.config.bos_token_id
            self.model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_from_pretrained, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_from_pretrained, padding_side='left')
        if self.peft_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, self.peft_path, config=self.peft_config)
        self.model_to_device()
        self.model = self.model.eval()

    def model_to_device(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model.to(self.device)
        self.true_model = self.model.module if hasattr(
            self.model, 'module') else self.model
    
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
        
        if self.model_type == 'qwen2_5_vl':
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(history_list)
            return image_inputs, video_inputs


    def prepare_generate(self, query: str | list = '', history: List = None, build_message=True):
        if not isinstance(query, list):
            query = [query]
            history = [history] if history is not None else None
        
        with torch.no_grad():
            image_inputs = None
            video_inputs = None
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
                image_inputs, video_inputs = self.process_mm(query, history)
            
            if image_inputs is not None or video_inputs is not None:
                batched_inputs = self.processor(
                    text=inputs,
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True).to(self.device)
            else:
                batched_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True).to(self.device)

            if self.model_type == 'llama':
                batched_inputs = batched_inputs.data
            
            return batched_inputs
    
    def predict(self, query: str | list = '', history: List = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True):
        """
        Predicts the response for the given query and history.

        Args:
            query (str | list): The input query or a list of queries.
            history (List): The conversation history.
            max_new_tokens (int): Maximum number of new tokens to generate.
            num_beams (int): Number of beams for beam search.
            top_p (float): Top-p sampling parameter.
            temperature (float): Temperature for sampling.
            do_sample (bool): Whether to use sampling or greedy decoding.
            build_message (bool): Whether to build the chat input message.

        Returns:
            List[str]: The generated responses.
        """
        batched_inputs = self.prepare_generate(query, history, build_message)
        batched_outputs = self.true_model.generate(**batched_inputs, **{
                'max_new_tokens': max_new_tokens,
                'num_beams': num_beams,
                'do_sample': do_sample,
                'top_p': top_p,
                "temperature": temperature,
                "eos_token_id": self.eos_token_id
            })
        batched_response = self.process_model_outputs(batched_inputs, batched_outputs, self.tokenizer)
        return batched_response

    def __call__(self, query: str | list = '', history: List = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True):
        return self.predict(query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)
    
    def predict_stream(self, query: str | list = '', history: List = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = True, build_message=True):
        """
        Generates responses for the given query and history in a streaming manner.
        
        Args:
            query (str | list): The input query or a list of queries.
            history (List): The conversation history.
            max_new_tokens (int): Maximum number of new tokens to generate.
            num_beams (int): Number of beams for beam search.
            top_p (float): Top-p sampling parameter.
            temperature (float): Temperature for sampling.
            do_sample (bool): Whether to use sampling or greedy decoding.
            build_message (bool): Whether to build the chat input message.
        
        Yields:
            Tuple[List[str], List[str]]: A tuple containing the new text generated and the accumulated outputs.
        """

        batched_inputs = self.prepare_generate(query, history, build_message)

        streamer = BatchTextIteratorStreamer(len(batched_inputs['input_ids']), self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **batched_inputs,
            'max_new_tokens': max_new_tokens,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'top_p': top_p,
            "temperature": temperature,
            "eos_token_id": self.eos_token_id,
            'streamer': streamer
        }
        
        def thread_func():
            self.true_model.generate(**generation_kwargs)
        
        generation_thread = threading.Thread(target=thread_func)
        generation_thread.start()
        
        outputs = ["" for _ in range(len(batched_inputs['input_ids']))]
        for new_text in streamer:
            for idx, new_next_item in enumerate(new_text):
                if new_next_item is None:
                    continue
                outputs[idx] += new_next_item
            yield new_text, outputs


class BatchTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, batch_size:int, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)