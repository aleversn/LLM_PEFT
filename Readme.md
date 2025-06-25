## 本项目用于LLM-PEFT入门使用

本项目旨在提供一个简明的入门指南，帮助用户基于主流大模型（如 Llama3、GLM4、Qwen2）进行参数高效微调（PEFT）训练和推理。

## 🛠️ 一、环境配置

### ✅ 安装依赖

#### - 支持 Llama3、GLM4、Qwen2 模型：

```bash
pip install protobuf transformers>=4.44.1 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

> ChatGLM3-6B和GLM4可能存在`transformers`版本限制, 注意降级.

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

#### - 支持VL模型推理

```python
pred([{
    "role": "user",
    "content": [
        {"type": "image", "image": "./example.jpg"},
        {"type": "text", "text": "她是谁?"},
    ]
}, {
    "role": "user",
    "content": [
        {"type": "image", "image": "./example.jpg"},
        {"type": "text", "text": "她有哪些著名作品?"},
    ]
}])
```

针对不同模型, 请在[vLLM 文档](https://docs.vllm.ai/en/latest/examples/offline_inference/vision_language.html)上详见参数配置, 并直接在Predictor中设置.

---

### 2. 项目封装推理调用

#### - 示例：LLM推理

```python
from main.predictor.llm import Predictor

predictor = Predictor(model_from_pretrained="model/chatglm3-6b")
res = predictor("你好?", history=[])
print(res)
```

- history: history为二维数组, 其中每一项对应一个`query`的`history`.

#### - 支持VL模型推理

```python
pred([{
    "role": "user",
    "content": [
        {"type": "image", "image": "./example.jpg"},
        {"type": "text", "text": "她是谁?"},
    ]
}, {
    "role": "user",
    "content": [
        {"type": "image", "image": "./example.jpg"},
        {"type": "text", "text": "她有哪些著名作品?"},
    ]
}])
```

其中, 对于大尺寸图片需指定其最大像素, 设置方法如下 (以Qwen2.5-VL为例, `最大像素N`一般设为`N*28*28`):

```python
# min_pixels and max_pixels
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# resized_height and resized_width
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "min_pixels": 50176,
                "max_pixels": 50176,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

#### - 支持流式推理：

```python
for res in predictor.predict_stream("你的任务是什么?", history=[]):
    print(res[1])
```

---

### 3. LoRA 微调模型推理

#### - 示例：ChatGLM LoRA 推理

```python
from main.predictor.llm import Predictor

predictor = Predictor(model_from_pretrained="model/chatglm3-6b", peft_path='<PEFT_PATH>')
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

本项目现已支持`ChatGLM3`、`ChatGLM4`、`Qwen2.5`、`Llama3`等系列的模型进行PEFT+PPO微调训练，使用时注意使用上述模型对应的transformers版本，推荐使用如下版本：
| 模型系列        |推荐transformers版本                      |
| -----------  | ----------------------------- |
| ChatGLM3     |  `4.40.0`   |
| ChatGLM4     |  `>=4.46.0` （如需要使用`>=4.49.0`，需到[huggingface](https://huggingface.co/THUDM/glm-4-9b-chat/commit/bd8234fe5e0c09c48637a92abb0c797cb5fa0e73)上更新`modeling_chatglm.py`文件）  |
| Qwen2.5      |  `4.43.0`   |
| Llama3/3.1/3.2      |  `4.43.0`   |
```python
from main.trainer.chatglm_rlhf_base import Trainer
from transformers import AutoTokenizer, AutoConfig
import datetime

tokenizer = AutoTokenizer.from_pretrained(r"/root/autodl-tmp/models/chatglm4-9b-chat", trust_remote_code=True)
config = AutoConfig.from_pretrained(r"/root/autodl-tmp/models/chatglm4-9b-chat", trust_remote_code=True)

trainer = Trainer(tokenizer=tokenizer,
 config=config, 
 from_pretrained=r"/root/autodl-tmp/models/chatglm4-9b-chat", 
 reward_from_pretrained=r"/root/autodl-tmp/models/text2vec-base-multilingual", 
 loader_name='LLM_RLHF',
 data_path='Wiki_Humans_RL_100', 
 ratio_for_rlhf=-1.0, 
 max_length=1024, 
 batch_size=4, 
 task_name='Wiki-humans-rawrl-' + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

for i in trainer(num_epochs=50, weight_for_cos_and_jaccard=[0.5, 0.5], ppo_epsilon=0.15, lr=5e-4, ppo_epochs=3, alpha=0.5, beta=0.5, gamma=0): 
    a = i
```

基本设置：
- `reward_from_pretrained`: Reward Model模型文件，在本项目中使用轻便、能准确分词的非通用模型即可实现训练（如`text2vec`、`qwen3-embedding`等）
- `loader_name`: 数据加载器的名称，可在`main/loaders.py`下查看当前支持的数据形式
- `data_path`: 数据地址，你需要先创建一个`present.json`，在该文件下进行路径指定，具体操作方法前面已提到，注意，你需要到`loaders.py`中将`data_path`修改存放`present.json`的位置
- `ratio_for_rlhf`: 进行在线强化学习的概率，可以设置为完全在线学习(=1)，或完全离线学习(<=0)
- `actor_resume_path`: 策略模型预训练文件(可选)
- `critic_resume_path`: 评论家模型预训练文件(可选)

其他关键设置：
- `weight_for_cos_and_jaccard`: 权重矩阵，分配奖励分数中cos相似度指标与jaccard相似度指标的权重，注意二者之和要为1；如果你感兴趣，可以根据任务需求在`model`下修改奖励分配方法
- `ppo_epsilon`: PPO裁切系数
- `ppo_epoch`: 指定重要性采样次数，也就是参考模型需要在策略模型更新多少次后进行更新
- `alpha`、`beta`、`gamma`:分别确定PPO损失式中，策略模型损失、评论家模型损失、熵的权重

数据格式：

- `LLM_RLHF`: 训练数据集格式包含`conversations`, `gold_answers`和`bad_answers`三个字段.

```json
{
  "conversations": [...],
  "gold_answers": ["理想答案"],
  "bad_answers": ["错误答案1", "错误答案2"]
}
```

- PPO对参数极为敏感，且可调节的参数数量较多，因此训练时要多次尝试，找到表现较好的参数组合

- 为了方便对训练进行监控，PPO训练引入了`tensorboard`进行性能监控，可以通过`tensorboard`面板查看当前模型的训练情况
    - 首先安装`tensorboard`
    ```bash
        pip install tensorboard
    ```
    - 开始训练后，可在终端使用如下指令打开`tensorboard`面板
    ```bash
        tensorboard --logdir={your_saved_dir} [--port=xx]
    ```
    其中`--logdir`是你保存的`tensorboard`文件的地址，本项目默认保存在`logs/tensorboard_logs`下；`--port`可以指定面板加载的端口，若不指定，默认在`localhost:6006`上打开。
    - `tensorboard`仅在单卡训练时会有比较直观的监控效果，多卡时曲线会重叠，建议跑多卡前先在单卡上用`tensorboard`看一下效果，然后再在多卡上正式跑

- 若你有新的想法，需要对上述数据格式进行修改，并修改`loaders`、`models`、`trainers`下的文件，以适配你的设定

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
from main.predictor.llm import Predictor

pred = Predictor(model_from_pretrained='./model/chatglm3-6b', peft_path='./save_model/RAG/ChatGLM_44136')

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
result = pred(rag_user_question, history=history)
print(result[0])
```

---

## ⏫ 七、辅助的验证集/测试集批量推理

```python
from main.evaluation.inferences import inference_with_data_path
from main.predictor.llm import Predictor

pred = Predictor(model_from_pretrained='/home/lpc/models/chatglm3-6b/', peft_path='./save_model/ChatGLM_LoRA')

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