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

def inference_with_data(data, batcher, save_path=None, batch_size=1, skip=-1, yield_output=False):
    '''
    - `params`: `data`: list, data is the dataset to inference.
    - `params`: `batcher`: function, batcher is the function to batch the data.
    - `params`: `save_path`: str, save_path is the path to save the inference result.
    - `params`: `batch_size`: int, batch_size is the batch size of the inference.
    - `params`: `skip`: int, skip is the number of data to skip.
    - `params`: `yield_output`: bool, yield_output is the flag to yield the output.
    - `return`: `result`: list, result is the inference result.
    '''
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    result = []
    selected_data = data[skip:] if skip > 0 else data
    num_batches = len(selected_data) // batch_size + (1 if len(selected_data) % batch_size != 0 else 0)
    for idx in tqdm(range(num_batches)):
        sample = selected_data[idx * batch_size: (idx + 1) * batch_size]
        inputs = {
            'query': [item['query'] for item in sample],
            'history': [item['history'] for item in sample]
        }
        output = batcher(inputs)
        if isinstance(output, list):
            result.extend(output)
            for out in output:
                if save_path is not None:
                    with open(save_path, encoding='utf-8', mode='a') as f:
                        f.write(json.dumps(out, ensure_ascii=False) + '\n')
            if yield_output:
                format_output = []
                for i in range(len(output)):
                    format_output.append({
                        'idx': idx * batch_size + i,
                        'query': sample[i]['query'],
                        'history': sample[i]['history'],
                        'response': output[i]
                    })
                yield format_output
        else:
            result.append(output)
            if save_path is not None:
                with open(save_path, encoding='utf-8', mode='a') as f:
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
            if yield_output:
                format_output = {
                    'idx': idx,
                    'query': sample[0]['query'],
                    'history': sample[0]['history'],
                    'response': output
                }
                yield format_output
        
    yield result

def inference_with_data_path(data_path, batcher, save_path, batch_size=1, skip=-1, eval_mode='dev'):
    '''
    - `params`: `data_path`: str, data_path is the dataset source, which is defined in `./data/present.json`.
    - `params`: `batcher`: function, batcher is the function to batch the data.
    - `params`: `save_path`: str, save_path is the path to save the inference result.
    - `params`: `batch_size`: int, batch_size is the batch size of the inference.
    - `params`: `skip`: int, skip is the number of data to skip.
    - `params`: `eval_mode`: str, eval_mode is the mode of the inference, which is defined in `./data/present.json`.
    - `return`: `result`: list, result is the inference result.
    '''
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
            
    for result in inference_with_data(data=format_data, batcher=batcher, save_path=save_path, batch_size=batch_size, skip=skip):
        return result