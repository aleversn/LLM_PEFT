## 本项目用于LLM-PEFT入门使用

本项目旨在提供一个简明的入门指南，帮助用户基于主流大模型（如 Llama3、GLM4、Qwen2）进行参数高效微调（PEFT）训练和推理。

## 🛠️ 一、环境配置

### ✅ 安装依赖

#### - 支持 Llama3、GLM4、Qwen2 模型：

```bash
pip install protobuf transformers>=4.44.1 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

#### - 若使用 GLM3 模型：

```bash
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

---

### ✅ vLLM 推理加速（推荐）

建议使用 vLLM 以提升推理效率。

#### - 创建 Conda 环境：

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

#### - 安装 vLLM（需 CUDA >= 12.1）：

```bash
pip install vllm
```

更多安装方式详见 [vLLM 官网](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)

---

## ✨ 二、模型推理使用说明

### 1. 使用 vLLM 进行推理：

```python
from main.predictor.vllm import Predictor

pred = Predictor(model_from_pretrained='Qwen/Qwen3-8B')
result = pred('采购人委托采购代理机构代理采购项目，发布招标公告后，有权更换采购代理机构吗?', max_new_tokens=512)
print(result)
```

---

### 2. 项目封装推理调用

根据模型类型选择对应模块：

| 模型类型        | 使用模块                     |
| ----------- | ------------------------ |
| ChatGLM（≤3） | `main.predictor.chatglm` |
| 其他          | `main.predictor.llm`     |

#### - 示例：ChatGLM 推理

```python
from main.predictor.chatglm import Predictor

predictor = Predictor(model_name="ChatGLM2-6B", model_from_pretrained="model/chatglm3-6b")
res = predictor("你好?", history=[])
print(res)
```

#### - 支持流式推理：

```python
for res in predictor.stream_chat("你的任务是什么?", history=[]):
    sys.stdout.write('\r' + res[0])
    sys.stdout.flush()
```

---

### 3. LoRA 微调模型推理

| 模型类型        | 使用模块                          |
| ----------- | ----------------------------- |
| ChatGLM（≤3） | `main.predictor.chatglm_lora` |
| 其他          | `main.predictor.llm_lora`     |

#### - 示例：ChatGLM LoRA 推理

```python
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='./model/chatglm3-6b', resume_path='./save_model/RAG/ChatGLM_44136')
result = pred('采购人委托采购代理机构代理采购项目，发布招标公告后，有权更换采购代理机构吗?', max_new_tokens=512)
print(result)
```

---

### 4. Transformers 官方推理方法

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).half().cuda()
model.eval()
```

#### - 直接推理：

```python
response, history = model.chat(tokenizer, "你的任务是什么?", history=[])
print(response)
```

#### - 流式推理：

```python
for response, history in model.stream_chat(tokenizer, "你的任务是什么?", history=[]):
    print(response)
```

---

## 🔥 三、PEFT 微调训练

```python
from main.trainer.llm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/chatglm3-6b", trust_remote_code=True)

trainer = Trainer(
    tokenizer=tokenizer,
    config=config,
    from_pretrained='./model/chatglm3-6b',
    loader_name='ChatGLM_Chat',
    data_path='<dataset_name>',
    max_new_tokens=3600,
    batch_size=1,
    task_name='<dataset_name>'
)
```

- loader_name: 数据集加载器, 其中`ChatGLM <= 3`为`ChatGLM_Chat`, 其余均使用`LLM_Chat`.
- `<dataset_name>`: 表示选用的训练数据集类型, 请创建`./data/present.json`文件并自定义数据集路径.

### 数据集配置说明

请在 `./data/present.json` 中配置训练数据路径：

```json
{
  "qa_dataset": {
    "train": "./data/qa_train.json",
    "dev": "./data/qa_dev.json"
  },
  "law_dataset": {
    "train": "./data/law_train.json",
    "dev": "./data/law_dev.json"
  }
}
```

当此时, `<dataset_name>`选取为`qa_dataset`时, 模型将自动读取对应的`train`, `dev`和`test`(可缺省)路径下的数据集.

### 数据格式

* **ChatGLM\_Chat 格式**（推荐）：

```json
{"conversations": [{"role": "user", "content": "请识别xxx\n输入: 三件事不能硬撑"}, {"role": "assistant", "content": "好的, 答案是xxx"}]}
```

* **ChatGLM\_LoRA 格式**（更灵活）：

```json
[{"context": "Instruction: 请识别xxx\n输入: 三件事不能硬撑\nAnswer: ", "target": "好的, 答案是xxx\n"}]
```

---

## 🚀 四、分布式训练支持

### ✅ 使用 Accelerate 分布式训练：

```bash
accelerate launch --num_processes=<n_gpu> <your_script>.py
```

> 注意：batch\_size 表示每个 GPU 上的 batch 大小。

---

### ✅ 启用 DeepSpeed ZeRO-3 + 张量并行

#### 安装：

```bash
pip install deepspeed
```

#### 配置：

```bash
accelerate config
# DeepSpeed -> Yes
# DeepSpeed config file -> ./ds_config.json
```

- 配置过程中, GPU选择`multi-GPU`

- 是否使用DeepSpeed: `Yes`

- 是否指定DeepSpeed配置文件: `Yes`

- DeepSpeed配置文件路径: `./ds_config.json`

其他选项默认为`No`, 即[yes/No]中直接回车.

`ds_config.json`配置文件内容如下:

示例配置文件 `ds_config.json`：

```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        }
    },
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "fp16": {
        "enabled": true
    }
}
```

---

## 🎭 五、PEFT + PPO 强化学习微调

```python
from main.trainer.chatglm_rlhf import Trainer

trainer = Trainer(
    tokenizer=tokenizer,
    config=config,
    from_pretrained='/home/lpc/models/chatglm3-6b/',
    reward_from_pretrained='/home/lpc/models/text2vec-base-chinese/',
    loader_name='ChatGLM_RLHF',
    data_path='ID',
    max_new_tokens=1200,
    batch_size=2,
    task_name='ID'
)

for i in trainer(num_epochs=5):
    a = i
```

- `reward_from_pretrained`: Reward Model模型文件

数据格式：

- `ChatGLM_RLHF`: 训练数据集格式包含`conversations`, `gold_answers`和`bad_answers`三个字段.

```json
{
  "conversations": [...],
  "gold_answers": ["理想答案"],
  "bad_answers": ["错误答案1", "错误答案2"]
}
```

---

## 💭 六、RAG（检索增强生成）推理

使用前, 需安装好`chromadb`

```bash
pip install chromadb
```

### ✅ 构建 chromadb 检索数据库

```python
import chromadb
from chromadb.utils import embedding_functions

DB_SAVE_DIR = './数据库目录'
DB_NAME = '读取的数据库名称'
N_RESULTS = 1

client = chromadb.PersistentClient(DB_SAVE_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="DMetaSoul/sbert-chinese-general-v2")
collection = client.get_or_create_collection(DB_NAME, embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})
```

### ✅ 启用 RAG 推理：

```python
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='./model/chatglm3-6b', resume_path='./save_model/RAG/ChatGLM_44136')

user_question = '这里是用户的提问'
# 检索相关片段
res = collection.query(
    query_texts=[user_question],
    n_results=N_RESULTS
)
# 根据距离判断是否引用检索信息, 如果检索片段与目标距离过大(即相关性低), 则不使用clue
if len(res['metadatas'][0]) > 0:
    distance = res['distances'][0][0]
    if distance < 0.1:
        clue = res['metadatas'][0][0]['clue']
    else:
        clue = False
else:
    clue = False
if not clue:
    rag_user_question = user_question
else:
    rag_user_question = f'<rag>检索增强知识: \n{clue}</rag>\n请根据以上检索增强知识回答以下问题\n{user_question}'
# 拼接好线索后进行提问
result = pred.chat(rag_user_question, history=history)
history = result[1]
print(result[0])
```

---

## ⏫ 七、辅助的验证集/测试集批量推理

```python
from main.evaluation.inferences import inference_with_data_path
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='/home/lpc/models/chatglm3-6b/', resume_path='./save_model/ChatGLM_LoRA')

def batcher(item):
    return pred(**item, max_new_tokens=1024, temperature=0, build_message=True)

inference_with_data_path(data_path='YOUR_PATH', batcher=batcher, save_path='./outputs.txt', batch_size=4)
```

---

## 🧪 八、评估指标

### - 单条文本评估：

```python
from main.evaluation.metrics import evaluate_all_metrics

# 测试示例
reference_text = ["I love this cat.", "I really love this cat."]
generated_text = "hahaha I love this cat."

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
scores = evaluate_all_metrics(tokenizer, reference_text, generated_text, intensive=False) # 如果是中文请将intensive设置为True
print(scores)
```

### - 批量评估：

```python
from main.evaluation.metrics import evaluate_generation

# 测试示例
reference_text = ["I love this cat.", "I really love this cat."]
generated_text = "hahaha I love this cat."

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
scores = evaluate_generation(tokenizer, [reference_text], [generated_text], intensive=False) # 如果是中文请将intensive设置为True
print(scores)
```

```bash
+--------+--------+--------+--------+--------+---------+---------+---------+---------+---------+--------+--------+
| Metric | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-3 | ROUGE-4 | ROUGE-L | METEOR |  TER   |
+--------+--------+--------+--------+--------+---------+---------+---------+---------+---------+--------+--------+
| Scores |  0.7   | 0.6236 | 0.5298 | 0.4518 |  0.8889 |  0.8571 |   0.8   |  0.6667 |  0.8889 | 0.958  | 0.3043 |
+--------+--------+--------+--------+--------+---------+---------+---------+---------+---------+--------+--------+

{'BLEU-1': 0.7, 'BLEU-2': 0.6236095644623235, 'BLEU-3': 0.5297521706139517, 'BLEU-4': 0.4518010018049224, 'ROUGE-1': 0.888888888888889, 'ROUGE-2': 0.8571428571428571, 'ROUGE-3': 0.8, 'ROUGE-4': 0.6666666666666666, 'ROUGE-L': 0.888888888888889, 'METEOR': 0.9579668787425148, 'TER': 0.30434782608695654}
```