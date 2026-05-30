import time
from typing import Dict
import os


import torch
import pickle
from loguru import logger
from redis import Redis
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..state import ResponseState


class SemanticCache:

    def __init__(self, embedding_model: HuggingFaceEmbedding, cache_similarity_threshold: float = 0.85):
        
        self.embedding_model = embedding_model
        self.cache_similarity_threshold = cache_similarity_threshold
        logger.info("loading redis....")
        self.redis = Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
        )
        self.cache: Dict[str, dict] = {}
        self._load_from_redis()


    def _load_from_redis(self):

        for key in self.redis.keys("cache:*"): # list of keys in 'binary' 
            _key = key.decode().removeprefix("cache:") # <- getting keys from redis
            _value = pickle.loads(self.redis.get(key))  # <- getting values from redis
            self.cache[_key] = _value


    def get(self, state: ResponseState) -> tuple:

        rewritten_query = state.get("rewritten_query")

        embedded_query = torch.tensor(
            self.embedding_model.get_text_embedding(
                rewritten_query
        ))

        if not self.cache:
            # cache is empty
            return {"cache_hit": None, "embedded_query": embedded_query}
        
        for cached_query, info in self.cache.items():
            cached_emb = info.get("embedding")
            similarity = torch.nn.functional.cosine_similarity(
                embedded_query,
                cached_emb,
                dim=0
            )

            if similarity >= self.cache_similarity_threshold:
                # cache match
                return {"cache_hit": True, 
                        "answer": info.get("answer")}


        # cache no match
        return {"cache_hit": False, "embedded_query": embedded_query}

    def store(self, state: ResponseState):
        
        rewritten_query = state.get("rewritten_query")
        answer = state.get("answer")
        embedded_query = state.get("embedded_query")

        _key = rewritten_query 
        _value ={
            "answer": answer,
            "embedding": embedded_query,
            "timestamp": time.time()
            } 
        
        self.cache[_key] = _value #for current runtime

        #for persistent storage
        self.redis.set( 
            f"cache:{_key}",  # adding prefix. Redis has built-in conversion for bytes
            pickle.dumps(_value) # <- converting to binary before storing to redis
        ) 

        return {}