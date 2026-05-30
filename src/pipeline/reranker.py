import time

from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List

from ..state import ResponseState

class Reranker():

    def __init__(self, reranking_model: SentenceTransformerRerank):
        self.reranking_model = reranking_model
        

    def rerank(self, state: ResponseState) -> dict:
        rewritten_query = state.get("rewritten_query")
        retrieved_nodes = state.get("retrieved_nodes")

        t0 = time.perf_counter()
        reranked_nodes = self.reranking_model.postprocess_nodes(
            retrieved_nodes, 
            query_str=rewritten_query
        )
        t1 = round(time.perf_counter() - t0, 2) 

        top_relevancy_score = reranked_nodes[0].score
    
        return {"reranked_nodes": reranked_nodes,
                "reranking_time": t1,
                "top_relevancy_score": top_relevancy_score}
