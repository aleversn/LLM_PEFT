import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from functools import partial


class CriticModel(nn.Module):
    def __init__(self, model_from_pretrained, resume_path=None, layers_keep=1, device='cuda') -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_from_pretrained, trust_remote_code=True)
        self.config.num_layers = layers_keep
        model = AutoModel.from_pretrained(
            model_from_pretrained, trust_remote_code=True, config=self.config)
        model = model.transformer
        # solve RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
        if "cuda" in device:
            model = model.half().cuda(device)  # half for gpu only
        elif "cpu" == device:
            model = model.bfloat16()
        else:
            model = model.float()
        self.model = model
        self.output_linear = nn.Linear(
            self.config.hidden_size, 1, device=self.model.device, dtype=self.model.dtype)
        self.dtype = self.model.dtype
        self.device = self.model.device
        if resume_path is not None:
            self.output_linear.load_state_dict(torch.load(resume_path))

    def forward(self, **kwargs):
        output = self.model(**kwargs)
        values = torch.tanh(self.output_linear(output.last_hidden_state))
        return values.transpose(0, 1).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, model_from_pretrained, device="cuda") -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(
            model_from_pretrained)
        model = AutoModel.from_pretrained(model_from_pretrained)
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def jaccard(s1, s2):
        assert len(s1)+len(s2) > 0
        s1 = set(s1)
        s2 = set(s2)
        s_or = s1 | s2
        s_and = s1 & s2
        jaccard_distance = len(s_and)/len(s_or)
        return jaccard_distance

    def forward(self, gen_texts=["I am the generated content from LLM."],
                gold_answers=['Generation from LLM.', "I am the output content from LLM."],
                bad_answers=['I am human.', 'I am the content from human writing.'],
                weight_for_cos_and_jaccard=[0.5, 0.5]):
        examples = gold_answers + bad_answers
        example_num = len(examples)
        assert len(gen_texts) > 0 and example_num > 0
        reward_direction = torch.ones(example_num, device=self.model.device)
        reward_direction[len(gold_answers):] = -1
        sentences = gen_texts + examples
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, return_tensors='pt')
        ids = self.tokenizer.batch_encode_plus(
            sentences, add_special_tokens=False)["input_ids"]
        # temporary truncate position_ids
        batch_size, max_seq_len = encoded_input["input_ids"].shape
        if max_seq_len > self.model.config.max_position_embeddings:
            encoded_input["position_ids"] = torch.arange(
                max_seq_len).expand((1, -1)).repeat(batch_size, 1)
            encoded_input["position_ids"] = encoded_input["position_ids"] / \
                max_seq_len*self.model.config.max_position_embeddings
            encoded_input["position_ids"] = encoded_input["position_ids"].floor(
            ).long()
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = encoded_input.to(self.model.device)
            model_output = self.model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])
        gen_text_vecs = sentence_embeddings[:len(gen_texts)]
        answers_vecs = sentence_embeddings[len(gen_texts):]
        reward_ = []
        for i in range(gen_text_vecs.shape[0]):
            gen_text_vecs_ = gen_text_vecs[i:i+1]
            # 用一下广播计算cos
            coses = torch.cosine_similarity(
                gen_text_vecs_, answers_vecs, dim=1)
            # 余弦截断
            coses[(coses < 0)] = 0
            # 计算 jaccard距离
            jaccard_s1 = partial(RewardModel.jaccard, ids[i])
            jaccards = torch.tensor(np.vectorize(jaccard_s1)(np.array(
                ids[-len(examples):], dtype=object)), dtype=coses.dtype, device=coses.device)
            similarity = weight_for_cos_and_jaccard[0] * \
                coses + weight_for_cos_and_jaccard[1]*jaccards
            value, index = similarity.max(dim=-1)
            reward_.append(value*reward_direction[index])
        reward = torch.stack(reward_)
        return reward
