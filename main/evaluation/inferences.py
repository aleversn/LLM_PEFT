import os
import json
from tqdm import tqdm

def get_data_present(present_path):
    if not os.path.exists(present_path):
        return {}
    with open(present_path, encoding='utf-8') as f:
        present_json = f.read()
    data_present = json.loads(present_json)
    return data_present

def inference_with_data_path(data_path, batcher, save_path, skip=-1, eval_mode='dev'):
    data_present = get_data_present('./data/present.json')
    file_name = data_present[data_path][eval_mode]
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    format_data = []
    for item in data:
        item = item['conversations'] if 'conversations' in item else item
        assert len(item) > 1
        if len(item) >= 2:
            query = item[-2]['content']
            history = item[:-2]
        else:
            query = item[-1]['content']
            history = []
        format_data.append({'query': query, 'history': history})
            
    return inference_with_data(format_data, batcher, save_path, skip)

def inference_with_data(data, batcher, save_path, skip=-1):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = []
    for idx, item in enumerate(tqdm(data)):
        if idx < skip:
            continue
        output = batcher(item)
        result.append(output)
        with open(save_path, encoding='utf-8', mode='a') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
    return result
