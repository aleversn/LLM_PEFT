### 本项目用于LLM-PEFT入门使用

#### 前置工作

安装环境

```bash
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

#### 推理

1. Transformers官方方法

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()
history = []
```

- 1.1 直接推理

```python
response, history = model.stream_chat(tokenizer, "你的任务是什么?", history=history)
print(response)
```

- 1.2 流式推理

```python
for response, history in model.stream_chat(tokenizer, "你的任务是什么?", history=history):
    print(response)
```

2. 项目封装方法

```python
import sys
from main.predictor.chatglm import Predictor

predictor = Predictor(model_name="ChatGLM2-6B", model_from_pretrained="model/chatglm3-6b")
```

- 2.1 直接推理

```python
res = predictor("你好?", history=[])
print(res)
```

- 2.2 流式推理

```python
history = []
for res in predictor.stream_chat("你的任务是什么?", history=history):
    sys.stdout.write('\r' + res[0])
    sys.stdout.flush()
```

3. PEFT模型推理

```python
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='./model/chatglm3-6b', resume_path='./save_model/RAG/ChatGLM_44136')
```

- 3.1 直接推理

```python
result = pred('采购人委托采购代理机构代理采购项目，发布招标公告后，有权更换采购代理机构吗?', max_length=512)
print(result)
```

- 3.2 流式推理

```python
history = []
result = pred.chat('采购人委托采购代理机构代理采购项目，发布招标公告后，有权更换采购代理机构吗?', max_length=3000, history=history)
history = result[1]
print(result[0])
```

#### PEFT微调训练

```python
from main.trainer.chatglm_lora import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='./model/chatglm3-6b', loader_name='ChatGLM_Chat', data_path='<dataset_name></dataset_name>', max_length=3600, batch_size=1, task_name='<dataset_name>')
```

- `<dataset_name>`: 表示选用的训练数据集类型, 请创建`./data/present.json`文件并自定义数据集路径, 例如:

`present.json`

```json
{
    "qa_dataset": {
        "train": "./data/...",
        "dev": "./data/..."
    },
    "law_dataset": {
        "train": "./data/...",
        "dev": "./data/..."
    }
}
```

当此时, `<dataset_name>`选取为`qa_dataset`时, 模型将自动读取对应的`train`, `dev`和`test`(可缺省)路径下的数据集.

- `loader_name`: 表示使用的数据装载器, 目前开发了`ChatGLM_LoRA`和`ChatGLM_Chat`两种.

**数据集格式**

- `ChatGLM_LoRA`: 训练数据集格式与GLM一致,样例为:

```json
[{"role": "user", "content": "请识别xxx\n输入: 三件事不能硬撑"}, {"role": "assistant", "content": "好的, 答案是xxx"}]
[{"role": "user", "content": "指令: 请识别xxx\n输入: 问答"}, {"role": "assistant", "content": "好的, 答案是xxx"}]
[{"role": "user", "content": "指令: 请识别xxx\n输入: 节奏"}, {"role": "assistant", "content": "好的, 答案是xxx"}]
```

亦或是

```json
{"conversations": [{"role": "user", "content": "请识别xxx\n输入: 三件事不能硬撑"}, {"role": "assistant", "content": "好的, 答案是xxx"}]}
{"conversations": [{"role": "user", "content": "指令: 请识别xxx\n输入: 问答"}, {"role": "assistant", "content": "好的, 答案是xxx"}]}
{"conversations": [{"role": "user", "content": "指令: 请识别xxx\n输入: 节奏"}, {"role": "assistant", "content": "好的, 答案是xxx"}]}
```

- `ChatGLM_LoRA`: 为自定义的输入输出格式, 形式上更加自由, 样例为:

```json
[{"context": "Instrcution: 请识别xxx\n输入: 三件事不能硬撑\n Answer: ", "target": "好的, 答案是xxx\n"}]
```

#### PEFT + PPO训练

```python
from main.trainer.chatglm_rlhf import Trainer
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
config = AutoConfig.from_pretrained("/home/lpc/models/chatglm3-6b/", trust_remote_code=True)
trainer = Trainer(tokenizer=tokenizer, config=config, from_pretrained='/home/lpc/models/chatglm3-6b/', reward_from_pretrained='/home/lpc/models/text2vec-base-chinese/', loader_name='ChatGLM_RLHF', data_path='ID', max_length=1200, batch_size=2, task_name='ID')

for i in trainer(num_epochs=5):
    a = i
```

- `reward_from_pretrained`: Reward Model模型文件

**数据集格式**

- `ChatGLM_RLHF`: 训练数据集格式包含`conversations`, `gold_answers`和`bad_answers`三个字段.

```json
{"conversations": [{"role": "user", "content": "你的主人是谁？"}, {"role": "assistant", "content": "张三是我的主人。"}], "gold_answers": ["张三是我的主人。"], "bad_answers": ["我没有主人", "我不知道", "我没有真正的主人", "我是人工智能没有主人"]}
```

#### RAG推理

使用前, 需安装好`chromadb`

```python
# 创建或者加载chromadb客户端
import chromadb
from chromadb.utils import embedding_functions

DB_SAVE_DIR = './数据库目录'
DB_NAME = '读取的数据库名称'
N_RESULTS = 1

client = chromadb.PersistentClient(DB_SAVE_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="DMetaSoul/sbert-chinese-general-v2")
collection = client.get_or_create_collection(DB_NAME, embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})
```

加载模型

```python
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='./model/chatglm3-6b', resume_path='./save_model/RAG/ChatGLM_44136')
```

```python
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

#### 验证集/测试集生成推理

建议采用`Predictor`中的默认方法, 以便支持批量生成.

```python
from main.evaluation.inferences import inference_with_data_path
from main.predictor.chatglm_lora import Predictor

pred = Predictor(model_from_pretrained='/home/lpc/models/chatglm3-6b/', resume_path='./save_model/ChatGLM_LoRA')

def batcher(item):
    return pred(**item, max_length=1024, temperature=0, build_message=True)

inference_with_data_path(data_path='YOUR_PATH', batcher=batcher, save_path='./outputs.txt', batch_size=4)
```

你可以自行实现`batcher`仅需确保返回的是生成文本即可,
若你希望能够自行喂入数据, 也可以使用`inference_with_data`, 注意每一条格式为`{"query": "", "history": []}`

#### 评估性能

- 单例计算

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

- 批量计算

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