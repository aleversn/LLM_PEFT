import asyncio
from concurrent.futures import ThreadPoolExecutor

from main.predictor.llm import Predictor

class LLMAPI:
    def __init__(self, model_from_pretrained, peft_path=None, batch_size=20):
        self.model_from_pretrained = model_from_pretrained
        self.peft_path = peft_path
        self.batch_size = batch_size
        self.llm_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pred = Predictor(model_from_pretrained=self.model_from_pretrained, peft_path=self.peft_path)

    def get_predictor(self):
        return self.pred
    
    async def lm_chat(self, query: str | list = '', history = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True,
            **kwargs):
    
        loop = asyncio.get_event_loop() # 获取当前运行事件循环

        def run_async_pred(query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message):
            try:
                if isinstance(query, str):
                    query = [query]
                if history is None:
                    history = []
                num_batch = len(query) // self.batch_size + 1 if  len(query) % self.batch_size != 0 else len(query) // self.batch_size
                results = []
                for i in range(num_batch):
                    q_list = query[i * self.batch_size: (i + 1) * self.batch_size]
                    h_list = history[i * self.batch_size: (i + 1) * self.batch_size]
                    results.extend(self.pred.predict(q_list, h_list, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message))
                return results
            except:
                return ['error' for _ in query]

        async with self.llm_lock:
            return await loop.run_in_executor(self.executor, run_async_pred, query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)  # 在线程池中运行预测函数


    async def lm_stream_chat(self, query: str | list = '', history = None, max_new_tokens=512, num_beams:int=1, top_p: float = 0.8, temperature=1.0, do_sample: bool = False, build_message=True,
                **kwargs):
        
        loop = asyncio.get_event_loop() # 获取当前运行事件循环
        queue = asyncio.Queue()

        def run_async_pred_stream(query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message):
            try:
                for result in self.pred.predict_stream(query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message):
                    asyncio.run_coroutine_threadsafe(queue.put(result), loop) # 利用事件循环将结果放入队列
            finally:
                asyncio.run_coroutine_threadsafe(queue.put('stop'), loop)
        
        async with self.llm_lock:
            loop.run_in_executor(self.executor, run_async_pred_stream, query, history, max_new_tokens, num_beams, top_p, temperature, do_sample, build_message)  # 在线程池中运行预测函数

            while True:
                result = await queue.get()
                if result == 'stop':
                    break
                yield result