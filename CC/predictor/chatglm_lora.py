from transformers import AutoModel
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, PeftModelForCausalLM
from typing import Tuple, List
from copy import deepcopy
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
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
    
    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        for response in output.split("<|assistant|>"):
            metadata, content = response.split("\n", maxsplit=1)
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    def tool_call(**kwargs):
                        return kwargs
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history
    
    @torch.inference_mode()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to(self.device)
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.get_command("<|user|>"),
                        self.tokenizer.get_command("<|observation|>")]
        outputs = self.true_model.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = self.tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history

    def __call__(self, text='', max_length=150, temperature=0):
        return self.predict(text=text, max_length=max_length, temperature=temperature)

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores