import torch
from typing import TypedDict, List, Optional, Annotated
from llama_index.core.schema import NodeWithScore
import operator

class ResponseState(TypedDict):
    query : str
    rewritten_query: str
    cache_hit: Optional[bool]
    embedded_query: torch.Tensor
    retrieved_nodes: List[NodeWithScore]
    reranked_nodes: List[NodeWithScore]
    top_relevancy_score: float
    compressed_nodes: List[NodeWithScore]
    answer: str
    retrieval_time: float
    reranking_time: float
    generation_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: int
    rewrite_retry_count: Annotated[int, operator.add]
