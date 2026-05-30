import time

from loguru import logger
from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core import VectorStoreIndex
from ..core.text_utils import TextUtils
from ..state import ResponseState


class Retriever:

    def __init__(self, index: VectorStoreIndex, top_k: int = 20):
        self.textutils = TextUtils()
        self.index = index
        self.top_k = top_k

        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k) # <- LlamaIndex embeddes query internally
    
    
    def retrieve(self, state: ResponseState) -> dict:

        rewritten_query = state.get("rewritten_query")
        normalized_query = self.textutils.normalize(query=rewritten_query)
        
        t0 = time.perf_counter()
        retrieved_nodes = self.retriever.retrieve(normalized_query)
        t1 = round(time.perf_counter() - t0, 2)
        
        if len(retrieved_nodes) == 0:
            raise ValueError(f"no nodes retrieved for query....")
        
        return {"retrieved_nodes": retrieved_nodes,
                "retrieval_time": t1}