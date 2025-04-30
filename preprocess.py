# %%
import json
from tqdm import tqdm

filename = '/home/lpc/repos/SAS_Benchmark/GaoKao_Subjective_BE/backend_data/scores/7_Math_ShortAns.jsonl'
with open(filename) as f:
    ori_data = f.readlines()
ori_data = [json.loads(line) for line in ori_data]

result = []
for item in tqdm(ori_data):
    res_segs = item['bad_student_answer_segs']
    last_idx = 0
    format_response = {
        'id': item['id'],
        'question': item['question'],
        'reference': item['answer'],
        'analysis': item['analysis'],
        'total': item['score'],
        'steps': []
    }
    if 'scoreItem' not in item:
        result.append(format_response)
        continue
    scoreItem = item['scoreItem']
    format_response['manual_label'] = scoreItem['label']
    seg_labels = scoreItem['seg_labels']
    seg_labels = json.loads(seg_labels)
    
    for seg_item in seg_labels:
        seg_idx, seg_label, seg_errors = seg_item['idx'], seg_item['label'], seg_item['errors']
        if seg_label != '' and int(seg_label) == 0 and len(seg_errors) == 0:
            continue
        if seg_label != '' and int(seg_label) > 0 and len(seg_errors) == 0:
            seg_errors = ['步骤正确']
        format_response['steps'].append({
            'response': '\n'.join(res_segs[last_idx: seg_idx + 1]),
            'label': seg_label,
            'errors': seg_errors
        })
        last_idx = seg_idx + 1
    result.append(format_response)

with open('./datasets/7_Math_ShortAns.jsonl', 'w') as f:
    for item in result:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
