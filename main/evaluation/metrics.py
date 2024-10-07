# %%
'''
Requirements: pip install nltk rouge-score pyter3
'''
import nltk
import jieba
import pyter
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from rouge_chinese import Rouge

# 确保你安装了nltk库
# !pip install nltk

def calculate_bleu_scores(reference_tokens, generated_tokens):
    """
    计算 BLEU-1 到 BLEU-4 分数
    - `param` `reference_tokens`: 参考文本 (列表格式，包含多个参考句子)
    - `param` `generated_tokens`: 生成文本 (字符串格式)
    :return: BLEU-1 到 BLEU-4 分数
    """
    # 使用SmoothingFunction来避免BLEU为0的情况
    smoothing_function = SmoothingFunction().method1
    
    # BLEU-1 到 BLEU-4
    bleu_1 = sentence_bleu(reference_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu_2 = sentence_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu_3 = sentence_bleu(reference_tokens, generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu_4 = sentence_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
    
    return bleu_1, bleu_2, bleu_3, bleu_4

# 确保你安装了rouge-score库
# !pip install rouge-score

def calculate_rouge_scores(reference, generated):
    """
    计算 ROUGE-1 到 ROUGE-4 分数
    - `param` `reference`: 参考文本 (字符串格式)
    - `param` `generated`: 生成文本 (字符串格式)
    :return: 各类 ROUGE 分数
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rouge3'].fmeasure, scores['rouge4'].fmeasure, scores['rougeL'].fmeasure

def calculate_meteor_score(reference_tokens, generated_tokens):
    """
    计算 METEOR 分数
    - `param` `reference_tokens`: 参考文本 (字符串格式)
    - `param` `generated_tokens`: 生成文本 (字符串格式)
    :return: METEOR 分数
    """
    reference_tokens = [[str(i) for i in tokens] for tokens in reference_tokens]
    generated_tokens = [str(i) for i in generated_tokens]
    return meteor_score(reference_tokens, generated_tokens)

def calculate_ter_score(reference_tokens, generated_tokens):
    """
    计算 TER 分数
    - `param` `reference_tokens`: 参考文本 (字符串格式)
    - `param` `generated_tokens`: 生成文本 (字符串格式)
    :return: TER 分数
    """    
    return pyter.ter(reference_tokens, generated_tokens)

def evaluate_all_metrics(tokenizer, reference, generated, intensive=False):
    """
    计算 BLEU-1 到 BLEU-4, ROUGE-1 到 ROUGE-4, METEOR 和 TER 分数
    - `param` `tokenizer`: 分词器
    - `param` `reference`: 参考文本 (列表格式，包含多个参考句子)
    - `param` `generated`: 生成文本 (字符串格式)
    - `param` `intensive`: 是否为字符密集型语言 (例如中文)
    :return: 各类指标的分数
    """
    reference_tokens = tokenizer(reference)['input_ids']  # 分词
    generated_tokens = tokenizer(generated)['input_ids']  # 分词
    if intensive:
        reference_intensive = list(jieba.cut(reference[0]))
        generated_intensive = list(jieba.cut(generated))
        reference_intensive = ' '.join(reference_intensive)
        generated_intensive = ' '.join(generated_intensive)
    else:
        reference_intensive = reference[0]
        generated_intensive = generated
    # 计算 BLEU 分数
    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(reference_tokens, generated_tokens)
    
    # 计算 ROUGE 分数 (假设第一个参考句子为基准)
    if intensive:
        rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-l"])
        scores = rouge.get_scores(' '.join(generated_intensive), ' '.join(reference_intensive))
        rouge_1 = scores[0]['rouge-1']['f']
        rouge_2 = scores[0]['rouge-2']['f']
        rouge_3 = scores[0]['rouge-3']['f']
        rouge_4 = scores[0]['rouge-4']['f']
        rouge_l = scores[0]['rouge-l']['f']
    else:
        rouge_1, rouge_2, rouge_3, rouge_4, rouge_l = calculate_rouge_scores(reference_intensive, generated_intensive)
    
    # 计算 METEOR 分数
    meteor = calculate_meteor_score(reference, generated)
    
    # 计算 TER 分数 (假设第一个参考句子为基准)
    ter = calculate_ter_score(reference, generated)
    
    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-3': rouge_3,
        'ROUGE-4': rouge_4,
        'ROUGE-L': rouge_l,
        'METEOR': meteor,
        'TER': ter
    }

def evaluate_generation(tokenizer, predictions, references, intensive=False, print_table=True):
    """
    计算 BLEU-1 到 BLEU-4, ROUGE-1 到 ROUGE-4, METEOR 和 TER 分数
    - `param` `tokenizer`: 分词器
    - `param` `predictions`: 参考文本 (列表格式，包含多个参考句子)
    - `param` `references`: 生成文本 (字符串格式)
    - `param` `intensive`: 是否为字符密集型语言 (例如中文)
    - `param` `print_table`: 是否打印表格
    :return: 各类指标的分数
    """
    results = {
        'BLEU-1': 0,
        'BLEU-2': 0,
        'BLEU-3': 0,
        'BLEU-4': 0,
        'ROUGE-1': 0,
        'ROUGE-2': 0,
        'ROUGE-3': 0,
        'ROUGE-4': 0,
        'ROUGE-L': 0,
        'METEOR': 0,
        'TER': 0
    }
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        scores = evaluate_all_metrics(tokenizer, pred, ref, intensive)
        for metric, score in scores.items():
            results[metric] += score
    for metric in results:
        results[metric] /= len(predictions)
    
    if print_table:
        from prettytable import PrettyTable
        table = PrettyTable()

        table.field_names = ["Metric"] + [metric for metric in scores.keys()]
        table.add_row(["Scores"] + [round(score, 4) for score in scores.values()])
        
        # 打印表格
        print(table)
    return results

