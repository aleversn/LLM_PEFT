import os
import sys
import json
from fastapi import FastAPI, File, UploadFile, Header, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from CC.api.lm_exs.chatglm_lora_rag_api import *
from CC.api.utils import response_body, ChatItem

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

API_KEY = 'creatorsn.com'
tmp_chat_dict = {}


@app.get("/")
def home():
    info = {
        'server_name': 'ChatGLM_API',
        'version': '0.0.1',
    }
    res = response_body(message='CCChatGLM API', data=info)
    return res()


@app.post("/chat")
async def chat(chat: ChatItem, api_key=Header(None)):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        return res()
    result = lm_chat(chat.query, chat.history)
    res = response_body(data={
        'answer': result[0],
        'history': result[1]
    })
    return res()


@app.post("/create_stream")
def create_stream_chat(chat: ChatItem, api_key=Header(None)):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        return res()
    guid = str(uuid.uuid4())
    tmp_chat_dict[guid] = chat
    res = response_body(data={'stream_id': guid})
    return res()


def chat_stream_call(stream_id: str, api_key=Header):
    if not api_key or api_key != API_KEY:
        res = response_body(code=401, status='error', message='Unauthorized')
        yield 'data:' + res.text() + '\n\n'
    chat = tmp_chat_dict.get(stream_id)
    if chat:
        for result in lm_stream_chat(chat.query, chat.history):
            answer, history = result[0], result[1]
            res = response_body(data={
                'answer': answer,
                'status': 'loading'
            })
            yield 'data:' + res.text() + '\n\n'
        res = response_body(data={
            'answer': answer,
            'status': 'finished'
        })
        del tmp_chat_dict[stream_id]
        yield 'data:' + res.text() + '\n\n'
    else:
        res = response_body(code=404, status='error',
                            message='Stream not found')
        yield 'data:' + res.text() + '\n\n'


@app.get("/chat_stream")
async def chat_stream(stream_id: str, api_key: str):
    return StreamingResponse(chat_stream_call(stream_id, api_key), media_type='text/event-stream')
