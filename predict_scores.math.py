# %%
import os
import json
import random
import json_repair
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append("./")
cmd_args = True
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=0, help='n_gpu')
parser.add_argument('--skip', default=-1, help='skip the first n lines, the skip index is count from the start index of n-th chunks')
parser.add_argument('--file_dir', default='./datasets', help='file name')
parser.add_argument('--file_name', default='7_Math_ShortAns', help='file name of the dataset, you should make sure it contains `test.jsonl` file')
parser.add_argument('--llm_name', default='', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/glm-4-9b-chat/', help='model from pretrained')
parser.add_argument('--batch_size', default=20, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

llm_name = args.llm_name if args.llm_name != '' else args.save_type_name
if llm_name == 'GLM3':
    from main.predictor.chatglm import Predictor
elif llm_name == 'Deepseek':
    from main.predictor.openai import Predictor
else:
    from main.predictor.llm import Predictor

if llm_name not in ['Deepseek']:
    pred = Predictor(model_from_pretrained=args.model_from_pretrained)
else:
    with open('api_key.txt') as f:
        api_key = f.read().strip()
    pred = Predictor(api_key=api_key, base_url='https://api.deepseek.com')

# %%
SOURCE_FILE = os.path.join(args.file_dir, f'{args.file_name}.jsonl')
ERROR_TYPE_FILE = os.path.join(args.file_dir, 'error_type.jsonl')
SAVE_DIR = os.path.dirname(SOURCE_FILE) + f'_{args.save_type_name}_Scored'
basename = os.path.basename(SOURCE_FILE)
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_scored.jsonl')
BATCH_SIZE = args.batch_size

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

ID = args.file_name.split('_')[0]
with open(ERROR_TYPE_FILE, encoding='utf-8') as f:
    error_type_list = f.readlines()
error_type_list = [json.loads(item) for item in error_type_list]
error_type_item = []
score_guideline = ''
for item in error_type_list:
    if str(item['q_id']) == str(ID):
        score_guideline = item['guideline']
        error_type_item = item['errors']
        error_type_item.append({'name': '步骤正确', 'description': '该步骤正确'})
        break

# 读取 JSON 文件并过滤无效数据
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()
ori_data = [json.loads(item) for item in ori_data]

# %%
prompt_prefix = '''请作为数学学科评分专家，根据以下要求对学生的作答进行专业评估：

【评估任务】
依据题目信息、参考答案及评分指南，对学生的分步解答进行精细化评分，并输出结构化评分结果。

【评分指南】
{score_guideline}

【评估材料】
- 试题内容：{question}
- 题目分值：{total}
- 错因类型：{error_type}
- 标准答案：{reference}
- 解析说明：{analysis}
- 学生作答：{student_answer}

【评估流程和要求】
1. 分步解析：
   - 拆解学生作答的每个解题步骤
   - 对每个步骤独立评估：
     * 判断正误（'label'）
     * 如存在错误，从错因列表中选取1项或多项主因（'errors'）
   - 单步评估格式：{{'step_score': 单步分数, 'errors': [错因]}}

2. 综合评定：
   - 汇总各步骤得分计算总分
   - 给出整体评价（'label'）

3. 结果输出：
   - 采用标准JSON格式输出：
     {{
       'total': 总分,
       'pred_score': 评估总分数,
       'steps': [各步骤评估结果]
     }}
    - 'pred_score'必须在'total'范围内
    - 分步的'step_score'累积值也必须在0到'pred_score'范围内

请按照上述规范完成评分，并以`JSON`格式输出标准化的评估结果。'''

# %%
all_examples = []
ask_list = []

error_type = []
for error_item in error_type_item:
    error_type.append(error_item['name'])
error_type_content = json.dumps(error_type, ensure_ascii=False)

for response_item in tqdm(ori_data):
    id = response_item['id']
    question = response_item.get('question', '')
    reference = response_item.get('reference', '')
    analysis = response_item.get('analysis', '')
    total = response_item.get('total', '')
    manual_label = response_item.get('manual_label', '')
    steps = response_item.get('steps', '')

    reponse_content = []
    for s_idx, step in enumerate(steps):
        response = step['response']
        reponse_content.append(f'## Step {s_idx}. {response}')
    
    # 构建问答回合内容
    ask_content = prompt_prefix.format(
        question=question,
        total=total,
        score_guideline=score_guideline,
        error_type=error_type_content,
        reference=reference,
        analysis=analysis,
        student_answer=''.join(reponse_content)
    )
    ask_list.append((ask_content, id))


#%%
# 计算总批次数
if llm_name not in ['Deepseek']:
    num_batches = len(ask_list) // BATCH_SIZE + (1 if len(ask_list) % BATCH_SIZE != 0 else 0)

    # 分批次进行预测并保存结果
    for i in tqdm(range(num_batches)):
        batch = ask_list[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        prompts = [item[0] for item in batch]
        ids = [item[1] for item in batch]
        max_length = [len(item[0]) for item in batch]
        max_length.sort(reverse=True)
        max_new_tokens = max_length[0]
        outputs = pred(prompts, max_new_tokens=1024, build_message=True, do_sample=False)
        
        for res, id in zip(outputs, ids):
            res = res.replace('\n', '')
            res = res.replace(' ', '')
            with open(SAVE_FILE, 'a', encoding='utf-8') as f:
                f.write(id + '\t' + json.dumps(res, ensure_ascii=False) + '\n')
else:
    for ask_content, id in tqdm(ask_list):
        res = pred(ask_content, model='deepseek-chat')
        res = res[0]
        res = res.replace('\n', '')
        res = res.replace(' ', '')
        with open(SAVE_FILE, 'a', encoding='utf-8') as f:
            f.write(id + '\t' + json.dumps(res, ensure_ascii=False) + '\n')

#%%
