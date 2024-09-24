# 创建或者加载chromadb客户端
from main.predictor.chatglm_lora import Predictor
import chromadb
from chromadb.utils import embedding_functions

DB_SAVE_DIR = '../../chroma_data'
DB_NAME = 'FDQA'
N_RESULTS = 1

client = chromadb.PersistentClient(DB_SAVE_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="DMetaSoul/sbert-chinese-general-v2")
collection = client.get_or_create_collection(
    DB_NAME, embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})


pred = Predictor(model_from_pretrained='../../model/chatglm3-6b',
                 resume_path='../../save_model/FDRAG/ChatGLM_44136')


def lm_chat(query: str, history=None, role: str = "user",
            max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
            **kwargs):
    res = collection.query(
        query_texts=[query],
        n_results=N_RESULTS
    )
    if len(res['metadatas']) > 0:
        clue = res['metadatas'][0][0]['clue']
    else:
        clue = '没有相关知识'
    rag_user_question = f'<rag>检索增强知识: \n{clue}</rag>\n请根据以上检索增强知识回答以下问题\n{query}'
    return pred.chat(rag_user_question, history, role, max_length, num_beams, do_sample, top_p, temperature, logits_processor, **kwargs)


def lm_stream_chat(query: str, history=None, role: str = "user",
                   past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                   logits_processor=None, return_past_key_values=False, **kwargs):
    res = collection.query(
        query_texts=[query],
        n_results=N_RESULTS
    )
    if len(res['metadatas'][0]) > 0:
        distance = res['distances'][0][0]
        if distance < 0.1:
            clue = res['metadatas'][0][0]['clue']
        else:
            clue = False
    else:
        clue = False
    if not clue:
        rag_user_question = query
    else:
        rag_user_question = f'<rag>检索增强知识: \n{clue}</rag>\n请根据以上检索增强知识回答以下问题\n{query}'
    for result in pred.stream_chat(rag_user_question, history, role, past_key_values, max_length, do_sample, top_p, temperature, logits_processor, return_past_key_values, **kwargs):
        yield result
