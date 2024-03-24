from CC.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='../../model/chatglm3-6b',
                 resume_path='../../save_model/FDRAG/ChatGLM_44136')


def lm_chat(query: str, history=None, role: str = "user",
            max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
            **kwargs):
    return pred.chat(query, history, role, max_length, num_beams, do_sample, top_p, temperature, logits_processor, **kwargs)


def lm_stream_chat(query: str, history=None, role: str = "user",
                   past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                   logits_processor=None, return_past_key_values=False, **kwargs):
    for result in pred.stream_chat(query, history, role, past_key_values, max_length, do_sample, top_p, temperature, logits_processor, return_past_key_values, **kwargs):
        yield result
