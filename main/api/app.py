import os
import sys
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Header, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from main.api.lm_exs.basic_api import LLMAPI
from main.api.utils import response_body, ChatItem

API_KEY = 'creatorsn.com'
MODEL_FROM_PRETRAINED = '/home/lpc/models/glm-4-9b-chat/'  # 模型路径
PEFT_PATH = None  # 如果有peft模型路径可以设置
BATCH_SIZE = 5
MAX_NEW_TOKENS = 8192
DO_SAMPLE = True
TEMPERATURE = 0.6

llm_api = LLMAPI(
    model_from_pretrained=MODEL_FROM_PRETRAINED,
    peft_path=PEFT_PATH,
    batch_size=BATCH_SIZE
)

msg_lock = asyncio.Lock()
tmp_msg_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(chat_stream_worker())
    yield # 在 yield 前的代码会在应用 启动时执行，在 yield 后的代码会在应用 关闭时执行。
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

@app.get("/")
def home():
    info = {
        'server_name': 'CCLLM_PEFT_API',
        'version': '0.0.1',
    }
    res = response_body(message='CCLLM_PEFT API', data=info)
    return res()


@app.post("/chat")
async def chat(chat: ChatItem, api_key=Header(None)):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        return res()
    result = await llm_api.lm_chat(chat.query, chat.history, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=DO_SAMPLE, temperature=TEMPERATURE)
    res = response_body(data={
        'answer': result
    })
    return res()


@app.post("/create_stream")
def create_stream_chat(chat: ChatItem, api_key=Header(None)):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        return res()
    guid = str(uuid.uuid4())
    tmp_msg_dict[guid] = chat
    res = response_body(data={'stream_id': guid})
    return res()

async def chat_stream_worker():
    def get_waiting_list():
        result = []
        for stream_id, chat in tmp_msg_dict.items():
            if chat.status == 'loading':
                result.append((stream_id, chat))
        return result[:BATCH_SIZE]
    
    is_exec = False
    while True:
        waiting_list = get_waiting_list()
        if len(waiting_list) < BATCH_SIZE and not is_exec:
            await asyncio.sleep(3)
            is_exec = True
            continue
        else:
            is_exec = False
        if len(waiting_list) == 0:
            continue
        query_list = [chat.query for _, chat in waiting_list]
        history_list = [chat.history for _, chat in waiting_list]
        async for output in llm_api.lm_stream_chat(query_list, history_list, max_new_tokens=MAX_NEW_TOKENS,
                                                    do_sample=DO_SAMPLE, temperature=TEMPERATURE):
            for i, out in enumerate(output[1]):
                guid = waiting_list[i][0]
                tmp_msg_dict[guid].response = out
        for i, out in enumerate(output[1]):
            guid = waiting_list[i][0]
            tmp_msg_dict[guid].status = 'finished'


async def chat_stream_call(stream_id: str, api_key=Header):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        yield 'data:' + res.text() + '\n\n'
    chat = tmp_msg_dict.get(stream_id)
    if chat:
        cache_response = chat.response
        while chat.status == 'loading':
            if chat.response != cache_response:
                res = response_body(data={
                    'answer': chat.response,
                    'status': 'loading'
                })
                cache_response = chat.response
                yield 'data:' + res.text() + '\n\n'
            else:
                await asyncio.sleep(0.1)
        res = response_body(data={
            'answer': chat.response,
            'status': 'finished'
        })
        del tmp_msg_dict[stream_id]
        yield 'data:' + res.text() + '\n\n'
    else:
        res = response_body(code=404, status='error',
                            message='Stream not found')
        yield 'data:' + res.text() + '\n\n'

@app.get("/chat_stream")
async def chat_stream(stream_id: str, api_key: str):
    return StreamingResponse(chat_stream_call(stream_id, api_key), media_type='text/event-stream')
